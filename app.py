from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import streamlit as st

# ===== 기본 세팅 =====
st.set_page_config(page_title="Amazons (1P vs CPU)", layout="wide")

SIZE = 10
EMPTY, HUM, CPU, BLOCK = 0, 1, 2, 3
DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

EMO_HUM = "🔵"
EMO_CPU = "🟡"
EMO_BLK = "⬛"
EMO_EMP = "·"

@dataclass
class Move:
    fr: Tuple[int,int]
    to: Tuple[int,int]
    shot: Tuple[int,int]

Board = List[List[int]]

# ===== 유틸 =====
def in_bounds(r:int,c:int)->bool:
    return 0 <= r < SIZE and 0 <= c < SIZE

def clone(b:Board)->Board:
    return [row[:] for row in b]

def iter_ray(b:Board, r:int,c:int, dr:int,dc:int):
    nr, nc = r+dr, c+dc
    while in_bounds(nr,nc) and b[nr][nc]==EMPTY:
        yield (nr,nc)
        nr += dr; nc += dc

def piece_positions(b:Board, side:int)->List[Tuple[int,int]]:
    token = HUM if side==HUM else CPU
    return [(r,c) for r in range(SIZE) for c in range(SIZE) if b[r][c]==token]

def legal_dests_from(b:Board, r:int,c:int)->List[Tuple[int,int]]:
    out=[]
    for dr,dc in DIRS:
        out.extend(iter_ray(b,r,c,dr,dc))
    return out

def legal_shots_from(b:Board, r:int,c:int)->List[Tuple[int,int]]:
    return legal_dests_from(b,r,c)

def apply_move(b:Board, mv:Move, side:int)->Board:
    nb = clone(b)
    (r1,c1),(r2,c2),(rs,cs) = mv.fr, mv.to, mv.shot
    nb[r1][c1] = EMPTY
    nb[r2][c2] = side
    nb[rs][cs] = BLOCK
    return nb

def has_any_move(b:Board, side:int)->bool:
    return any(legal_dests_from(b,r,c) for r,c in piece_positions(b, side))

# ===== 평가/AI (간결 버전: 난이도에 따라 깊이/분기 제한) =====
def mobility(b:Board, side:int)->int:
    return sum(len(legal_dests_from(b,r,c)) for r,c in piece_positions(b, side))

def liberties(b:Board, side:int)->int:
    s=0
    for r,c in piece_positions(b, side):
        for dr,dc in DIRS:
            nr,nc=r+dr,c+dc
            if in_bounds(nr,nc) and b[nr][nc]==EMPTY:
                s+=1
    return s

def center_score(b:Board, side:int)->int:
    cx, cy = (SIZE-1)/2, (SIZE-1)/2
    tot=0
    for r,c in piece_positions(b, side):
        tot -= int(abs(r-cx)+abs(c-cy))
    return tot

def evaluate(b:Board)->int:
    # +면 CPU 유리, -면 사람 유리
    return (10*(mobility(b,CPU)-mobility(b,HUM))
            + 2*(liberties(b,CPU)-liberties(b,HUM))
            + (center_score(b,CPU)-center_score(b,HUM)))

def gen_moves_limited(b:Board, side:int, k_dest:int, k_shot:int, cap:int)->List[Move]:
    out=[]
    for r,c in piece_positions(b, side):
        dests=legal_dests_from(b,r,c)
        # 이동만 반영하여 휴리스틱 스코어
        scored=[]
        for tr,tc in dests:
            tmp = clone(b); tmp[r][c]=EMPTY; tmp[tr][tc]=side
            sc = mobility(tmp, side) - mobility(tmp, HUM if side==CPU else CPU)
            scored.append(((tr,tc), sc))
        scored.sort(key=lambda x:x[1], reverse=True)
        for (tr,tc),_ in scored[:k_dest]:
            tmp = clone(b); tmp[r][c]=EMPTY; tmp[tr][tc]=side
            shots = legal_shots_from(tmp,tr,tc)
            s2=[]
            for sr,sc in shots:
                tmp2 = clone(tmp); tmp2[sr][sc]=BLOCK
                s2.append(((sr,sc), mobility(tmp2, side)-mobility(tmp2, HUM if side==CPU else CPU)))
            s2.sort(key=lambda x:x[1], reverse=True)
            for (sr,sc),_ in s2[:k_shot]:
                out.append(Move((r,c),(tr,tc),(sr,sc)))
                if len(out)>=cap: return out
    return out

def search(b:Board, depth:int, a:int, bb:int, side:int, P:Dict[str,int])->int:
    if depth==0 or not has_any_move(b, side):
        if not has_any_move(b, side):
            return 10_000 if side==HUM else -10_000
        return evaluate(b)

    k_d = P[f"k_dest_d{depth}"]; k_s = P[f"k_shot_d{depth}"]; cap = P[f"cap_d{depth}"]
    moves = gen_moves_limited(b, side, k_d, k_s, cap)
    if not moves: return 10_000 if side==HUM else -10_000

    if side==CPU:
        best=-1_000_000
        for mv in moves:
            val = search(apply_move(b,mv,CPU), depth-1, a, bb, HUM, P)
            best = max(best,val); a=max(a,val)
            if bb<=a: break
        return best
    else:
        best=1_000_000
        for mv in moves:
            val = search(apply_move(b,mv,HUM), depth-1, a, bb, CPU, P)
            best = min(best,val); bb=min(bb,val)
            if bb<=a: break
        return best

def ai_move(b:Board, difficulty:int)->Optional[Move]:
    if difficulty<=3:
        depth=1
        P=dict(k_dest_d1=6+difficulty*3, k_shot_d1=5+difficulty*2, cap_d1=40+difficulty*20)
    elif difficulty<=6:
        depth=2
        P=dict(k_dest_d2=8+(difficulty-3)*2, k_shot_d2=6+(difficulty-3), cap_d2=40+10*(difficulty-3),
               k_dest_d1=10, k_shot_d1=8, cap_d1=80)
    else:
        depth=3; s=difficulty-6
        P=dict(k_dest_d3=5+s, k_shot_d3=4+s//2, cap_d3=18+4*s,
               k_dest_d2=9+s, k_shot_d2=7+s//2, cap_d2=42+8*s,
               k_dest_d1=10, k_shot_d1=8, cap_d1=80)

    root = gen_moves_limited(b, CPU, P[f"k_dest_d{depth}"], P[f"k_shot_d{depth}"], P[f"cap_d{depth}"])
    if not root: return None
    best=None; val_best=-1_000_000
    for mv in root:
        v = search(apply_move(b,mv,CPU), depth-1, -1_000_000, 1_000_000, HUM, P)
        if v>val_best: val_best=v; best=mv
    return best

# ===== 초기 보드 =====
def initial_board()->Board:
    b = [[EMPTY for _ in range(SIZE)] for _ in range(SIZE)]
    # 사람(백) d1,g1,a4,j4  => (9,3),(9,6),(6,0),(6,9)
    b[9][3]=HUM; b[9][6]=HUM; b[6][0]=HUM; b[6][9]=HUM
    # 컴퓨터(흑) a7,j7,d10,g10 => (3,0),(3,9),(0,3),(0,6)
    b[3][0]=CPU; b[3][9]=CPU; b[0][3]=CPU; b[0][6]=CPU
    return b

# ===== 상태 =====
def reset_game():
    st.session_state.board = initial_board()
    st.session_state.turn = HUM
    st.session_state.phase = "select"  # select -> move -> shoot
    st.session_state.sel_from = None
    st.session_state.sel_to = None
    st.session_state.legal = set()
    st.session_state.difficulty = st.session_state.get("difficulty", 5)
    # 하이라이트 상태
    st.session_state.last_human_move = None  # Move or None
    st.session_state.last_cpu_move = None
    st.session_state.last_shot_pos = None     # (r,c) or None
    st.session_state.highlight_to = None      # 현재 턴에서 방금 이동한 곳

if "board" not in st.session_state:
    reset_game()

# ===== 상단 UI =====
hdr_left, hdr_right = st.columns([1,1])
with hdr_left:
    st.title("Amazons (1P vs CPU)")
    st.caption("말을 퀸처럼 이동 → 도착한 자리에서 또 퀸처럼 화살(블록)을 발사해 빈칸을 막기. 상대가 더 이상 이동 못 하면 승리.")
with hdr_right:
    diff = st.slider("난이도 (1 쉬움 ··· 10 어려움)", 1, 10, st.session_state.get("difficulty",5))
    st.session_state.difficulty = diff
    c1,c2 = st.columns(2)
    if c1.button("새 게임", use_container_width=True):
        reset_game()
        st.rerun()
    if c2.button("되돌리기(1수)", use_container_width=True):
        hist: List[Board] = st.session_state.get("hist", [])
        if hist:
            st.session_state.board = hist.pop()
        st.rerun()
st.session_state.setdefault("hist", [])

board: Board = st.session_state.board

# ===== 렌더링 보조(표시) =====
def cell_label(r:int,c:int)->str:
    """기본 말 + 하이라이트(선택◉ / 방금 이동✓ / 최근 블록✳)"""
    cell = board[r][c]
    if cell==HUM: label = EMO_HUM
    elif cell==CPU: label = EMO_CPU
    elif cell==BLOCK: label = EMO_BLK
    else: label = EMO_EMP

    # 선택 표시(내가 선택한 말)
    if st.session_state.turn==HUM and st.session_state.sel_from == (r,c) and st.session_state.phase in ("move","shoot"):
        label += "◉"

    # 이번 턴 내가 방금 옮겨 놓은 자리(사격 전 단계)
    if st.session_state.highlight_to == (r,c):
        label += "✓"

    # 최근 양측 이동 도착 칸
    hm = st.session_state.last_human_move
    cm = st.session_state.last_cpu_move
    if hm and hm.to == (r,c): label += "✓"
    if cm and cm.to == (r,c): label += "✓"

    # 최근 블록
    if st.session_state.last_shot_pos == (r,c) and board[r][c]==BLOCK:
        label += "✳"

    return label

# ===== 클릭 처리 =====
def on_click(r:int,c:int):
    turn = st.session_state.turn
    if turn != HUM: return
    phase = st.session_state.phase

    if phase=="select":
        if board[r][c]==HUM:
            st.session_state.sel_from = (r,c)
            st.session_state.legal = set(legal_dests_from(board,r,c))
            st.session_state.phase = "move"
            st.rerun()

    elif phase=="move":
        if (r,c) in st.session_state.legal:
            fr = st.session_state.sel_from
            nb = clone(board)
            nb[fr[0]][fr[1]] = EMPTY
            nb[r][c] = HUM
            st.session_state.board = nb
            st.session_state.sel_to = (r,c)
            st.session_state.highlight_to = (r,c)   # 방금 옮긴 자리 표시
            st.session_state.legal = set(legal_shots_from(nb,r,c))
            st.session_state.phase = "shoot"
            st.rerun()

    elif phase=="shoot":
        if (r,c) in st.session_state.legal:
            # 확정
            st.session_state.board[r][c] = BLOCK
            st.session_state.last_shot_pos = (r,c)
            # 최근 사람 수 기록
            hm = Move(st.session_state.sel_from, st.session_state.sel_to, (r,c))
            st.session_state.last_human_move = hm
            # 초기화
            st.session_state.hist.append(clone(board))
            st.session_state.turn = CPU
            st.session_state.phase = "select"
            st.session_state.sel_from = None
            st.session_state.sel_to = None
            st.session_state.legal = set()
            st.session_state.highlight_to = None
            st.rerun()

# ===== 보드 렌더(모든 칸을 동일한 버튼으로) =====
st.subheader("보드")
st.caption(f"{EMO_HUM}=플레이어  {EMO_CPU}=컴퓨터  {EMO_BLK}=블록  (◉ 선택, ✓ 방금 이동, ✳ 최근 블록)")

for r in range(SIZE):
    cols = st.columns(SIZE)
    for c in range(SIZE):
        label = cell_label(r,c)
        key = f"cell_{r}_{c}"
        clickable = False
        if st.session_state.turn==HUM:
            if st.session_state.phase=="select" and board[r][c]==HUM:
                clickable=True
            elif st.session_state.phase in ("move","shoot") and (r,c) in st.session_state.legal:
                clickable=True

        pressed = cols[c].button(label, key=key, disabled=not clickable)
        if pressed and clickable:
            on_click(r,c)

# ===== 엔드 체크 & AI =====
def announce(msg:str, ok=True):
    color = "#16a34a" if ok else "#dc2626"
    st.markdown(
        f"<div style='padding:8px;border-radius:8px;background:{'#ecfdf5' if ok else '#fef2f2'};color:{color}'>{msg}</div>",
        unsafe_allow_html=True
    )

if st.session_state.turn==HUM:
    if not has_any_move(board,HUM):
        announce("컴퓨터 승리! (플레이어가 움직일 곳이 없음)", ok=False)
else:
    if not has_any_move(board,CPU):
        announce("플레이어 승리! (컴퓨터가 움직일 곳이 없음)")
    else:
        with st.spinner("컴퓨터 생각중..."):
            mv = ai_move(board, st.session_state.difficulty)
            if mv is None:
                announce("플레이어 승리! (컴퓨터가 움직일 곳이 없음)")
            else:
                st.session_state.hist.append(clone(board))
                st.session_state.board = apply_move(board, mv, CPU)
                st.session_state.last_cpu_move = mv           # 컴퓨터 이동 표시
                st.session_state.last_shot_pos = mv.shot      # 최근 블록 표시
                st.session_state.turn = HUM
                st.session_state.phase = "select"
                st.session_state.sel_from = None
                st.session_state.sel_to = None
                st.session_state.legal = set()
        st.rerun()
