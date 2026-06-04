import os
import re
import math
import time
import threading
import pyperclip
from pynput import keyboard
import pystray
from PIL import Image, ImageDraw

# ==========================================
# 1. 第一階段：英數字與注音符號映射 (已修正 ㄧ U+3127)
# ==========================================
BOPOMOFO_MAP = {
    '1': 'ㄅ', 'q': 'ㄆ', 'a': 'ㄇ', 'z': 'ㄈ', '2': 'ㄉ', 'w': 'ㄊ', 's': 'ㄋ', 'x': 'ㄌ',
    'e': 'ㄍ', 'd': 'ㄎ', 'c': 'ㄏ', 'r': 'ㄐ', 'f': 'ㄑ', 'v': 'ㄒ', '5': 'ㄓ', 't': 'ㄔ',
    'g': 'ㄕ', 'b': 'ㄖ', 'y': 'ㄗ', 'h': 'ㄘ', 'n': 'ㄙ', 'u': 'ㄧ', 'j': 'ㄨ', 'm': 'ㄩ',
    '8': 'ㄚ', 'i': 'ㄛ', 'k': 'ㄜ', ',': 'ㄝ', '9': 'ㄞ', 'o': 'ㄟ', 'l': 'ㄠ', '.': 'ㄡ',
    '0': 'ㄢ', 'p': 'ㄣ', ';': 'ㄤ', '/': 'ㄥ', '-': 'ㄦ', ' ': ' ',
    '3': 'ˇ', '4': 'ˋ', '6': 'ˊ', '7': '˙'
}

def is_bopomofo_typo(text):
    if re.search(r'[3467]', text): return True
    if re.search(r'[a-zA-Z][/;,\.]|[/;,\.][a-zA-Z]', text): return True
    return any(char in BOPOMOFO_MAP for char in text.lower())

def decode_to_bopomofo(text):
    return "".join([BOPOMOFO_MAP.get(char, char) for char in text.lower()])

# ==========================================
# 2. 第二階段：注音自動斷詞器
# ==========================================
def split_bopo_string(bopo_str):
    pattern = r'[ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙ]*[ㄧㄨㄩ]*[ㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ]*[ˊˇˋ˙]?'
    return [m for m in re.findall(pattern, bopo_str) if m and m.strip()]

# ==========================================
# 3. 第三階段：解析本地端 word.csv 詞庫與語意加權
# ==========================================
EMISSION_DICT = {}  
BIGRAM_COUNTS = {}  

def init_local_csv_database():
    global EMISSION_DICT
    csv_filename = "word.csv"
    
    if not os.path.exists(csv_filename):
        print(f"\n[錯誤] 找不到 {csv_filename}！")
        return

    print(f"[TypoGuard] 偵測到本地字庫 {csv_filename}，正在讀入記憶體...")
    count = 0
    with open(csv_filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                char, _, bpmf = parts[0].strip(), parts[1].strip(), parts[2].strip()
                if not bpmf or not char: continue
                
                if bpmf not in EMISSION_DICT:
                    EMISSION_DICT[bpmf] = []
                
                rank = len(EMISSION_DICT[bpmf])
                score = max(0.001, 1.0 / (rank + 1))  
                EMISSION_DICT[bpmf].append((char, score))
                count += 1
    print(f"[TypoGuard] 成功載入本地詞庫！共導入 {count} 個字位對照。")

    # 關鍵語意優化：大量加入高頻二元接續詞
    global BIGRAM_COUNTS
    BIGRAM_COUNTS = {
        ('我', '也'): 10.0, ('也', '不'): 8.0, ('不', '知'): 12.0, ('知', '道'): 15.0,
        ('在', '打'): 7.0, ('什', '麼'): 20.0, ('你', '好'): 18.0, ('沒', '有'): 14.0,
        ('問', '題'): 16.0, ('聽', '說'): 15.0, ('長', '輩'): 12.0, ('真', '的'): 14.0,
        ('我', '是'): 30.0,   # 強力修正「我市」
        ('有', '點'): 15.0,
        ('點', '不'): 12.0,
        ('不', '懂'): 20.0,
        ('為', '什'): 18.0,
        ('情', '有'): 25.0,   
        ('有', '獨'): 25.0,
        ('獨', '鐘'): 30.0,   # 強力修正「讀中」
        ('獨', '鍾'): 30.0
    }

# ==========================================
# 4. 第四階段：基於 HMM 的 Viterbi 智慧選字演算法
# ==========================================
def get_transition_weight(prev_word, curr_word):
    return BIGRAM_COUNTS.get((prev_word, curr_word), 1.0)

def viterbi_decode(syllables):
    if not syllables: return ""
    viterbi_matrix = [{}]
    
    first_candidates = EMISSION_DICT.get(syllables[0], [])
    if not first_candidates: return "".join(syllables)
        
    for word, emit_score in first_candidates:
        prob = math.log(emit_score) + math.log(get_transition_weight('START', word))
        viterbi_matrix[0][word] = (prob, [word])
        
    for t in range(1, len(syllables)):
        viterbi_matrix.append({})
        candidates = EMISSION_DICT.get(syllables[t], [])
        
        if not candidates:
            for prev_word, (prev_log_prob, path) in viterbi_matrix[t-1].items():
                viterbi_matrix[t][prev_word] = (prev_log_prob, path + [syllables[t]])
            continue
            
        for curr_word, emit_score in candidates:
            max_prob = -float('inf')
            best_path = []
            
            for prev_word, (prev_log_prob, path) in viterbi_matrix[t-1].items():
                current_log_prob = prev_log_prob + math.log(emit_score) + math.log(get_transition_weight(prev_word, curr_word))
                if current_log_prob > max_prob:
                    max_prob = current_log_prob
                    best_path = path + [curr_word]
            
            if best_path:
                viterbi_matrix[t][curr_word] = (max_prob, best_path)
                
    last_step = viterbi_matrix[-1]
    if not last_step: return "".join(syllables)
    best_final_word = max(last_step, key=lambda w: last_step[w][0])
    return "".join(last_step[best_final_word][1])

# ==========================================
# 5. 第五階段：全域熱鍵與一鍵自動化
# ==========================================
keyboard_controller = keyboard.Controller()

def trigger_conversion():
    print("\n[TypoGuard] 觸發一鍵轉換...")
    raw_text = pyperclip.paste().strip()
    print(f"[原始亂碼]: {raw_text}")
    
    if not raw_text: return
        
    if is_bopomofo_typo(raw_text):
        bopo_text = decode_to_bopomofo(raw_text)
        syllables = split_bopo_string(bopo_text)
        chinese_result = viterbi_decode(syllables)
        
        print(f"[解碼注音]: {bopo_text}")
        print(f"[智慧選字]: {chinese_result}")
        
        pyperclip.copy(chinese_result)
        time.sleep(0.05)
        
        keyboard_controller.press(keyboard.Key.ctrl)
        keyboard_controller.press('v')
        time.sleep(0.05)
        keyboard_controller.release('v')
        keyboard_controller.release(keyboard.Key.ctrl)
        print("[系統狀態]: 已自動原地覆蓋完成！")
    else:
        print("[系統狀態]: 判定為正常文本，跳過轉換。")

# ==========================================
# 6. 第六階段：Windows 系統托盤與常駐控制
# ==========================================
def create_tray_icon():
    image = Image.new('RGB', (64, 64), color=(41, 128, 185)) 
    d = ImageDraw.Draw(image)
    d.rectangle([(24, 14), (40, 50)], fill=(255, 255, 255))
    d.rectangle([(14, 14), (50, 26)], fill=(255, 255, 255))
    return image

def on_quit(icon, item):
    print("[TypoGuard] 正在關閉防錯助手...")
    icon.stop()
    os._exit(0) 

def run_tray_system():
    icon = pystray.Icon(
        "TypoGuard", 
        create_tray_icon(), 
        "TypoGuard 智慧防錯助手常駐中", 
        menu=pystray.Menu(pystray.MenuItem("結束 TypoGuard", on_quit))
    )
    icon.run()

# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print("🚀 【智慧注音防錯工具 (TypoGuard)】初始化啟動中...")
    print("==================================================")
    
    # 1. 載入本地大腦
    init_local_csv_database()
    
    # 2. 啟動鍵盤監聽
    hotkeys = keyboard.GlobalHotKeys({
        '<ctrl>+<shift>+b': trigger_conversion
    })
    hotkeys.start()
    print("[TypoGuard] 全域熱鍵 [Ctrl + Shift + B] 監聽已就緒。")
    print("[TypoGuard] 正在將圖示掛載至 Windows 系統托盤...")
    print("💡 (注意：下方圖示啟動後，終端機卡住是正常現象，代表程式已成功進入背景常駐。)")
    
    # 3. 啟動主執行緒系統托盤
    run_tray_system()