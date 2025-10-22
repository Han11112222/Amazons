from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import streamlit as st

# ================= 기본 세팅 =================
st.set_page_config(page_title="Cool Choi Amazons", layout="wide")

SIZE = 10
EMPTY, HUM, CPU, BLOCK = 0, 1, 2, 3
DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

# 이모지 (선턴=파랑, 후턴=라임)
EMO_HUM, EMO_CPU, EMO_BLK, EMO_EMP, EMO_MOVE, EMO_SHOT = "🔵","🟢","⬛","·","🟩","🟥"

# 인당 제한시간(초)
TIME_LIMIT = 10 * 60  # 10분

@dataclass
class Move:
    fr: Tuple[int,int]
    to: Tuple[int,int]
    shot: Tuple[int,int]

Board = List[List[int]]

# ----------------- 보드/규칙 -----------------
def in_bounds(r:int,c:int)->bool: return 0 <= r < SIZE and 0 <= c < SIZE
def clone(b:Board)->Board: return [row[:] for row in b]

def iter_ray(b:Board, r:int,c:int, dr:int,dc:int):
    nr, nc = r+dr, c+dc
    while in_bounds(nr,nc) and b[nr][nc]==EMPTY:
        yield (nr,nc); nr += dr; nc += dc

def piece_positions(b:Board, side:int)->List[Tuple[int,int]]:
    token = HUM if side==HUM else CPU
    return [(r,c) for r in range(SIZE) for c in range(SIZE) if b[r][c]==token]

def legal_dests_from(b:Board, r:int,c:int)->List[Tuple[int,int]]:
    out=[]; [out.extend(iter_ray(b,r,c,dr,dc)) for dr,dc in DIRS]; return out

def legal_shots_from(b:Board, r:int,c:int)->List[Tuple[int,int]]: return legal_dests_from(b,r,c)

def apply_move(b:Board, mv:Move, side:int)->Board:
    nb = clone(b); (r1,c1),(r2,c2),(rs,cs) = mv.fr, mv.to, mv.shot
    nb[r1][c1] = EMPTY; nb[r2][c2] = side; nb[rs][cs] = BLOCK
    return nb

def has_any_move(b:Board, side:int)->bool:
    return any(legal_dests_from(b,r,c) for r,c in piece_positions(b, side))

# ----------------- 간단 평가/AI -----------------
def mobility(b:Board, side:int)->int:
    return sum(len(legal_dests_from(b,r,c)) for r,c in piece_positions(b, side))

def liberties(b:Board, side:int)->int:
    s=0
    for r,c in piece_positions(b, side):
        for dr,dc in DIRS:
            nr,nc=r+dr,c+dc
            if in_bounds(nr,nc) and b[nr][nc]==EMPTY: s+=1
    return s

def center_score(b:Board, side:int)->int:
    cx=cy=(SIZE-1)/2; tot=0
    for r,c in piece_positions(b, side): tot -= int(abs(r-cx)+abs(c-cy))
    return tot

def evaluate(b:Board)->int:
    return 10*(mobility(b,CPU)-mobility(b,HUM)) + 2*(liberties(b,CPU)-liberties(b,HUM)) + (center_score(b,CPU)-center_score(b,HUM))

def gen_moves_limited(b:Board, side:int, k_dest:int, k_shot:int, cap:int)->List[Move]:
    out=[]
    for r,c in piece_positions(b, side):
        dests=legal_dests_from(b,r,c)
        scored=[]
        for tr,tc in dests:
            tmp = clone(b); tmp[r][c]=EMPTY; tmp[tr][tc]=side
            scored.append(((tr,tc), mobility(tmp, side)-mobility(tmp, HUM if side==CPU else CPU)))
        scored.sort(key=lambda x:x[1], reverse=True)
        for (tr,tc),_ in scored[:k_dest]:
            tmp = clone(b); tmp[r][c]=EMPTY; tmp[tr][tc]=side
            s2=[]
            for sr,sc in legal_shots_from(tmp,tr,tc):
                tmp2 = clone(tmp); tmp2[sr][sc]=BLOCK
                s2.append(((sr,sc), mobility(tmp2, side)-mobility(tmp2, HUM if side==CPU else CPU)))
            s2.sort(key=lambda x:x[1], reverse=True)
            for (sr,sc),_ in s2[:k_shot]:
                out.append(Move((r,c),(tr,tc),(sr,sc)))
                if len(out)>=cap: return out
    return out

def search(b:Board, depth:int, a:int, bb:int, side:int, P:Dict[str,int])->int:
    if depth==0 or not has_any_move(b, side):
        if not has_any_move(b, side): return 10_000 if side==HUM else -10_000
        return evaluate(b)
    k_d=P.get(f"k_dest_d{depth}",8); k_s=P.get(f"k_shot_d{depth}",6); cap=P.get(f"cap_d{depth}",40)
    moves = gen_moves_limited(b, side, k_d, k_s, cap)
    if not moves: return 10_000 if side==HUM else -10_000
    if side==CPU:
        best=-1_000_000
        for mv in moves:
            val = search(apply_move(b,mv,CPU), depth-1, a, bb, HUM, P)
            best=max(best,val); a=max(a,val)
            if bb<=a: break
        return best
    else:
        best=1_000_000
        for mv in moves:
            val = search(apply_move(b,mv,HUM), depth-1, a, bb, CPU, P)
            best=min(best,val); bb=min(bb,val)
            if bb<=a: break
        return best

def ai_params_by_difficulty(d:int)->Tuple[int,Dict[str,int]]:
    if d<=3:  return 1, dict(k_dest_d1=6+d*3, k_shot_d1=5+d*2, cap_d1=40+d*20)
    if d<=6:
        x=d-3
        return 2, dict(k_dest_d2=8+2*x, k_shot_d2=6+x, cap_d2=40+10*x,
                       k_dest_d1=10, k_shot_d1=8, cap_d1=80)
    if d<=10:
        s=d-6
        return 3, dict(k_dest_d3=5+s, k_shot_d3=4+s//2, cap_d3=18+4*s,
                       k_dest_d2=9+s, k_shot_d2=7+s//2, cap_d2=42+8*s,
                       k_dest_d1=10, k_shot_d1=8, cap_d1=80)
    t=d-10
    return 4, dict(k_dest_d4=4+(t//2), k_shot_d4=3+(t//3), cap_d4=14+2*t,
                   k_dest_d3=6+(t//1), k_shot_d3=4+(t//2), cap_d3=20+3*t,
                   k_dest_d2=8+(t//1), k_shot_d2=6+(t//2), cap_d2=36+4*t,
                   k_dest_d1=10, k_shot_d1=8, cap_d1=80)

def ai_move(b:Board, difficulty:int)->Optional[Move]:
    depth, P = ai_params_by_difficulty(difficulty)
    root = gen_moves_limited(b, CPU, P.get(f"k_dest_d{depth}",8), P.get(f"k_shot_d{depth}",6), P.get(f"cap_d{depth}",40))
    if not root: return None
    best=None; val_best=-1_000_000
    for mv in root:
        v = search(apply_move(b,mv,CPU), depth-1, -1_000_000, 1_000_000, HUM, P)
        if v>val_best: val_best=v; best=mv
    return best

# ----------------- 초기 보드/상태 -----------------
def initial_board()->Board:
    b = [[EMPTY for _ in range(SIZE)] for _ in range(SIZE)]
    b[9][3]=HUM; b[9][6]=HUM; b[6][0]=HUM; b[6][9]=HUM
    b[3][0]=CPU; b[3][9]=CPU; b[0][3]=CPU; b[0][6]=CPU
    return b

def reset_game():
    st.session_state.board = initial_board()
    st.session_state.turn = HUM
    st.session_state.phase = "select"
    st.session_state.sel_from = None
    st.session_state.sel_to = None
    st.session_state.legal = set()
    st.session_state.difficulty = st.session_state.get("difficulty", 6)
    st.session_state.cell_px = 52
    st.session_state.last_human_move = None
    st.session_state.last_cpu_move = None
    st.session_state.last_shot_pos = None
    st.session_state.highlight_to = None
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.setdefault("hist", [])

    # 타이머
    st.session_state.hum_time = 0.0
    st.session_state.cpu_time = 0.0
    st.session_state.turn_start = time.time()  # 첫 턴 시작 시각(HUM)

if "board" not in st.session_state:
    reset_game()
else:
    # turn_start 누락 방지
    if "turn_start" not in st.session_state or st.session_state.turn_start is None:
        st.session_state.turn_start = time.time()

# ===== 타이머 유틸 =====
def _accumulate_time(side:int):
    """현재 턴 소요시간을 누적하고 turn_start 갱신"""
    now = time.time()
    elapsed = max(0.0, now - (st.session_state.turn_start or now))
    if side == HUM:
        st.session_state.hum_time += elapsed
    else:
        st.session_state.cpu_time += elapsed
    st.session_state.turn_start = now

def _current_used(side:int)->float:
    """표시용: 누적 + (진행중이면 실시간 가산)"""
    base = st.session_state.hum_time if side==HUM else st.session_state.cpu_time
    if st.session_state.turn == side and not st.session_state.game_over:
        now = time.time()
        base += max(0.0, now - (st.session_state.turn_start or now))
    return base

def _check_flagfall():
    """시간초과 체크"""
    if st.session_state.game_over: return
    if st.session_state.turn == HUM and _current_used(HUM) >= TIME_LIMIT:
        end_game("컴퓨터", human_win=False)
    elif st.session_state.turn == CPU and _current_used(CPU) >= TIME_LIMIT:
        end_game("플레이어", human_win=True)

def _switch_turn(to_side:int):
    """턴 전환(누적 후 전환)"""
    # 현재 턴의 시간을 누적
    _accumulate_time(st.session_state.turn)
    # 전환
    st.session_state.turn = to_side
    st.session_state.phase = "select"
    st.session_state.sel_from=None
    st.session_state.sel_to=None
    st.session_state.legal=set()
    st.session_state.highlight_to=None
    st.session_state.turn_start = time.time()

# ----------------- 상단 UI (타이머 포함) -----------------
left, right = st.columns([1,1])
with left:
    st.title("Cool Choi Amazons")
    # 타이머 표시
    hum_used = _current_used(HUM)
    cpu_used = _current_used(CPU)
    hum_rem = max(0, TIME_LIMIT - int(hum_used))
    cpu_rem = max(0, TIME_LIMIT - int(cpu_used))
    st.markdown(
        f"""
        **⏱ 누적시간**  
        - {EMO_HUM} 플레이어: **{int(hum_used//60):02d}:{int(hum_used%60):02d}** (잔여 {hum_rem//60:02.0f}:{hum_rem%60:02.0f})  
        - {EMO_CPU} 컴퓨터: **{int(cpu_used//60):02d}:{int(cpu_used%60):02d}** (잔여 {cpu_rem//60:02.0f}:{cpu_rem%60:02.0f})
        """
    )
with right:
    diff = st.slider("난이도 (1 쉬움 ··· 15 매우 어려움)", 1, 15, st.session_state.difficulty)
    st.session_state.difficulty = diff
    c1, c2 = st.columns(2)
    if c1.button("새 게임", use_container_width=True):
        reset_game(); st.rerun()
    if c2.button("되돌리기(1수)", use_container_width=True):
        if st.session_state.hist:
            st.session_state.board = st.session_state.hist.pop()
            # 되돌리기 후에는 사람 턴으로 간주
            st.session_state.turn = HUM
            st.session_state.phase = "select"
            st.session_state.turn_start = time.time()
        st.rerun()

# ----------------- 정사각형 보드 CSS -----------------
CELL = int(st.session_state.cell_px)
GAP  = 8
board_total_px = SIZE * CELL + (SIZE-1) * GAP  # 가로·세로 동일 (진짜 정사각형)

st.markdown(
    f"""
    <style>
      /* 보드 외곽 컨테이너: 진짜 정사각형 + 클릭 우선권(z-index) */
      .board-wrap {{
        width: {board_total_px}px;
        height: {board_total_px}px;
        margin: 6px auto 12px auto;
        padding: {GAP/2}px;
        border: 2px solid #94a3b8;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        position: relative;
        z-index: 2;
        background: white;
      }}
      .board-row .stColumns {{ gap: {GAP}px !important; }}
      .board-row div[data-testid="column"] {{ padding: 0 !important; }}
      .board-row {{ margin-bottom: {GAP}px; }}
      .board-row:last-child {{ margin-bottom: 0; }}
      .board-grid .stButton > button {{
        width: {CELL}px !important;
        height: {CELL}px !important;
        aspect-ratio: 1 / 1 !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: {CELL}px !important;
        border-radius: 10px !important;
        border: 1.5px solid #cbd5e1 !important;
        background: white !important;
        font-size: {int(CELL*0.45)}px !important;
        display: inline-flex; align-items: center; justify-content: center;
        position: relative;
        z-index: 3;
      }}
      .board-grid .stButton > button:disabled {{ opacity: 1.0 !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

board: Board = st.session_state.board

# ----------------- 렌더/입력 -----------------
def cell_label(r:int,c:int)->str:
    cell = board[r][c]
    label = EMO_EMP if cell==EMPTY else (EMO_HUM if cell==HUM else (EMO_CPU if cell==CPU else EMO_BLK))
    if not st.session_state.game_over and st.session_state.turn==HUM:
        if st.session_state.phase=="move" and (r,c) in st.session_state.legal and cell==EMPTY: label = EMO_MOVE
        elif st.session_state.phase=="shoot" and (r,c) in st.session_state.legal and cell==EMPTY: label = EMO_SHOT
    if st.session_state.turn==HUM and st.session_state.sel_from==(r,c) and st.session_state.phase in ("move","shoot"): label += "◉"
    if st.session_state.highlight_to == (r,c): label += "✓"
    hm = st.session_state.last_human_move; cm = st.session_state.last_cpu_move
    if hm and hm.to==(r,c): label += "✓"
    if cm and cm.to==(r,c): label += "✓"
    if st.session_state.last_shot_pos == (r,c) and cell==BLOCK: label += "✳"
    return label

def on_click(r:int,c:int):
    if st.session_state.game_over or st.session_state.turn!=HUM:
        return
    phase = st.session_state.phase
    if phase=="select":
        if board[r][c]==HUM:
            st.session_state.sel_from=(r,c)
            st.session_state.legal=set(legal_dests_from(board,r,c))
            st.session_state.phase="move"; st.rerun()
        return
    if phase=="move":
        if (r,c) in st.session_state.legal:
            fr = st.session_state.sel_from
            nb = clone(board); nb[fr[0]][fr[1]] = EMPTY; nb[r][c] = HUM
            st.session_state.board = nb
            st.session_state.sel_to = (r,c); st.session_state.highlight_to=(r,c)
            st.session_state.legal=set(legal_shots_from(nb,r,c))
            st.session_state.phase="shoot"; st.rerun()
        return
    if phase=="shoot":
        if (r,c) in st.session_state.legal:
            # 사람 턴 시간 누적 (사격으로 턴 종료 직전)
            _accumulate_time(HUM)
            st.session_state.board[r][c] = BLOCK
            st.session_state.last_shot_pos=(r,c)
            st.session_state.last_human_move = Move(st.session_state.sel_from, st.session_state.sel_to, (r,c))
            st.session_state.hist.append(clone(board))
            _switch_turn(CPU)  # 전환 및 turn_start 재설정
            st.rerun()
        return

# 안내
st.subheader("보드")
st.caption(f"{EMO_HUM}=플레이어(선턴)  {EMO_CPU}=컴퓨터(후턴)  {EMO_BLK}=블록  ({EMO_MOVE} 이동, {EMO_SHOT} 사격 · ◉ 선택 · ✓ 방금 이동 · ✳ 최근 블록)")

# 보드 렌더(항상 클릭 허용 → 유효성은 on_click에서 필터)
st.markdown('<div class="board-wrap"><div class="board-grid">', unsafe_allow_html=True)
for r in range(SIZE):
    st.markdown('<div class="board-row">', unsafe_allow_html=True)
    cols = st.columns(SIZE)
    for c in range(SIZE):
        if cols[c].button(cell_label(r,c), key=f"cell_{r}_{c}"):
            on_click(r,c)
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown("</div></div>", unsafe_allow_html=True)

# ----------------- 엔드체크 & AI -----------------
def end_game(winner_label: str, human_win: bool):
    st.session_state.game_over=True
    st.session_state.winner=winner_label
    if human_win: st.balloons()
    st.markdown(
        f"<div style='padding:10px;border-radius:8px;background:#f1f5f9;border:1px solid #cbd5e1'><b>{winner_label} 승리!</b></div>",
        unsafe_allow_html=True,
    )

def announce_and_set(who: str, ok=True):
    color = "#16a34a" if ok else "#dc2626"
    st.markdown(f"<div style='padding:8px;border-radius:8px;background:{'#ecfdf5' if ok else '#fef2f2'};color:{color}'><b>{who} 승리!</b></div>", unsafe_allow_html=True)

# 시간초과 먼저 체크
_check_flagfall()

if not st.session_state.game_over:
    if st.session_state.turn==HUM and not has_any_move(board,HUM):
        announce_and_set("컴퓨터", ok=False); end_game("컴퓨터", human_win=False)

if not st.session_state.game_over and st.session_state.turn==CPU:
    if not has_any_move(board,CPU):
        announce_and_set("플레이어", ok=True); end_game("플레이어", human_win=True)
    else:
        with st.spinner("컴퓨터 생각중..."):
            start_ai = time.time()
            mv = ai_move(board, st.session_state.difficulty)
            # AI 생각 시간 누적
            st.session_state.cpu_time += max(0.0, time.time() - start_ai)
            if mv is None:
                announce_and_set("플레이어", ok=True); end_game("플레이어", human_win=True)
            else:
                st.session_state.board = apply_move(board, mv, CPU)
                st.session_state.last_cpu_move = mv
                st.session_state.last_shot_pos = mv.shot
                _switch_turn(HUM)
        st.rerun()
