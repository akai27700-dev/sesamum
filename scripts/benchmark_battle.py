import numpy as np
import json
from pathlib import Path
from numba import njit
_FULL_MASK = np.uint64(18446744073709551615)

@njit(cache=True, fastmath=True, nogil=True)
def count_bits(x):
    c = np.int64(0)
    while x:
        c += np.int64(1)
        x &= x - np.uint64(1)
    return c

@njit(cache=True, fastmath=True, nogil=True)
def get_legal_moves(P, O):
    m, e, l = (np.uint64(35604928818740862), ~(P | O) & _FULL_MASK, np.uint64(0))
    t = P << np.uint64(1) & O & m
    t |= t << np.uint64(1) & O & m
    t |= t << np.uint64(1) & O & m
    t |= t << np.uint64(1) & O & m
    l |= t << np.uint64(1) & e
    t = P >> np.uint64(1) & O & m
    t |= t >> np.uint64(1) & O & m
    t |= t >> np.uint64(1) & O & m
    t |= t >> np.uint64(1) & O & m
    l |= t >> np.uint64(1) & e
    t = P << np.uint64(8) & O
    t |= t << np.uint64(8) & O
    t |= t << np.uint64(8) & O
    t |= t << np.uint64(8) & O
    l |= t << np.uint64(8) & e
    t = P >> np.uint64(8) & O
    t |= t >> np.uint64(8) & O
    t |= t >> np.uint64(8) & O
    t |= t >> np.uint64(8) & O
    l |= t >> np.uint64(8) & e
    t = P << np.uint64(7) & O & m
    t |= t << np.uint64(7) & O & m
    t |= t << np.uint64(7) & O & m
    t |= t << np.uint64(7) & O & m
    l |= t << np.uint64(7) & e
    t = P >> np.uint64(7) & O & m
    t |= t >> np.uint64(7) & O & m
    t |= t >> np.uint64(7) & O & m
    t |= t >> np.uint64(7) & O & m
    l |= t >> np.uint64(7) & e
    t = P << np.uint64(9) & O & m
    t |= t << np.uint64(9) & O & m
    t |= t << np.uint64(9) & O & m
    t |= t << np.uint64(9) & O & m
    l |= t << np.uint64(9) & e
    t = P >> np.uint64(9) & O & m
    t |= t >> np.uint64(9) & O & m
    t |= t >> np.uint64(9) & O & m
    t |= t >> np.uint64(9) & O & m
    l |= t >> np.uint64(9) & e
    return l & _FULL_MASK

@njit(cache=True, fastmath=True, nogil=True)
def get_flip(P, O, idx):
    m, mk, f = (np.uint64(1) << np.uint64(idx), np.uint64(35604928818740862), np.uint64(0))
    t = m << np.uint64(1) & O & mk
    t |= t << np.uint64(1) & O & mk
    t |= t << np.uint64(1) & O & mk
    t |= t << np.uint64(1) & O & mk
    t |= t << np.uint64(1) & O & mk
    if t << np.uint64(1) & P:
        f |= t
    t = m >> np.uint64(1) & O & mk
    t |= t >> np.uint64(1) & O & mk
    t |= t >> np.uint64(1) & O & mk
    t |= t >> np.uint64(1) & O & mk
    t |= t >> np.uint64(1) & O & mk
    if t >> np.uint64(1) & P:
        f |= t
    t = m << np.uint64(8) & O
    t |= t << np.uint64(8) & O
    t |= t << np.uint64(8) & O
    t |= t << np.uint64(8) & O
    if t << np.uint64(8) & P:
        f |= t
    t = m >> np.uint64(8) & O
    t |= t >> np.uint64(8) & O
    t |= t >> np.uint64(8) & O
    t |= t >> np.uint64(8) & O
    if t >> np.uint64(8) & P:
        f |= t
    t = m << np.uint64(7) & O & mk
    t |= t << np.uint64(7) & O & mk
    t |= t << np.uint64(7) & O & mk
    t |= t << np.uint64(7) & O & mk
    t |= t << np.uint64(7) & O & mk
    if t << np.uint64(7) & P:
        f |= t
    t = m >> np.uint64(7) & O & mk
    t |= t >> np.uint64(7) & O & mk
    t |= t >> np.uint64(7) & O & mk
    t |= t >> np.uint64(7) & O & mk
    t |= t >> np.uint64(7) & O & mk
    if t >> np.uint64(7) & P:
        f |= t
    t = m << np.uint64(9) & O & mk
    t |= t << np.uint64(9) & O & mk
    t |= t << np.uint64(9) & O & mk
    t |= t << np.uint64(9) & O & mk
    t |= t << np.uint64(9) & O & mk
    if t << np.uint64(9) & P:
        f |= t
    t = m >> np.uint64(9) & O & mk
    t |= t >> np.uint64(9) & O & mk
    t |= t >> np.uint64(9) & O & mk
    t |= t >> np.uint64(9) & O & mk
    t |= t >> np.uint64(9) & O & mk
    if t >> np.uint64(9) & P:
        f |= t
    return f & _FULL_MASK

@njit(cache=True, fastmath=True, nogil=True)
def evaluate_board(P, O, weights):
    empty = np.int64(64) - count_bits(P | O)
    score = np.float64(0.0)
    P_tmp = P
    while P_tmp:
        b = P_tmp & ~P_tmp + np.uint64(1)
        P_tmp ^= b
        idx = count_bits(b - np.uint64(1))
        if idx < len(weights):
            score += weights[idx]
    O_tmp = O
    while O_tmp:
        b = O_tmp & ~O_tmp + np.uint64(1)
        O_tmp ^= b
        idx = count_bits(b - np.uint64(1))
        if idx < len(weights):
            score -= weights[idx]
    progress = (64 - empty) / 64.0
    if empty >= 45:
        stage_weights = weights[64:80] if len(weights) > 80 else weights
    elif empty >= 15:
        stage_weights = weights[80:96] if len(weights) > 96 else weights
    else:
        stage_weights = weights[96:112] if len(weights) > 112 else weights
    mobility = count_bits(get_legal_moves(P, O)) - count_bits(get_legal_moves(O, P))
    if len(stage_weights) > 0:
        score += mobility * stage_weights[0] * 0.1
    return score

@njit(cache=True, fastmath=True, nogil=True)
def alphabeta(P, O, depth, alpha, beta, weights, passed):
    valid = get_legal_moves(P, O)
    if not valid:
        if passed:
            return np.float64(count_bits(P) - count_bits(O)) * 10000.0
        v = alphabeta(O, P, depth, -beta, -alpha, weights, True)
        return -v
    if depth <= 0:
        return evaluate_board(P, O, weights)
    moves = []
    v_temp = valid
    while v_temp:
        bit = v_temp & -v_temp
        v_temp ^= bit
        sq_idx = count_bits(bit - np.uint64(1))
        f = get_flip(P, O, sq_idx)
        new_P = (P | bit | f) & _FULL_MASK
        new_O = (O ^ f) & _FULL_MASK
        score = evaluate_board(new_P, new_O, weights)
        moves.append((bit, sq_idx, score))
    for i in range(len(moves)):
        for j in range(i + 1, len(moves)):
            if moves[j][2] > moves[i][2]:
                moves[i], moves[j] = (moves[j], moves[i])
    max_val = -1e+18
    for bit, sq_idx, _ in moves:
        f = get_flip(P, O, sq_idx)
        val = -alphabeta(O ^ f, P | bit | f, depth - 1, -beta, -alpha, weights, False)
        if val > max_val:
            max_val = val
        if max_val > alpha:
            alpha = max_val
        if alpha >= beta:
            break
    return max_val

def get_best_move(P, O, weights, depth=10):
    valid = get_legal_moves(P, O)
    if not valid:
        return -1
    moves = []
    v_temp = valid
    while v_temp:
        bit = v_temp & -v_temp
        v_temp ^= bit
        sq_idx = count_bits(bit - np.uint64(1))
        f = get_flip(P, O, sq_idx)
        new_P = (P | bit | f) & _FULL_MASK
        new_O = (O ^ f) & _FULL_MASK
        score = evaluate_board(new_P, new_O, weights)
        moves.append((bit, sq_idx, score))
    for i in range(len(moves)):
        for j in range(i + 1, len(moves)):
            if moves[j][2] > moves[i][2]:
                moves[i], moves[j] = (moves[j], moves[i])
    best_move = -1
    best_score = -1e+18
    for bit, sq_idx, _ in moves[:min(10, len(moves))]:
        f = get_flip(P, O, sq_idx)
        val = -alphabeta(O ^ f, P | bit | f, depth - 1, -1e+18, 1e+18, weights, False)
        if val > best_score:
            best_score = val
            best_move = sq_idx
    return best_move

def play_game(weights1, weights2, depth=10):
    B = np.uint64(34628173824)
    W = np.uint64(68853694464)
    turn = 1
    while True:
        if turn == 1:
            move = get_best_move(B, W, weights1, depth)
            if move == -1:
                if get_legal_moves(W, B) == 0:
                    break
                turn = -1
                continue
            f = get_flip(B, W, move)
            B = (B | np.uint64(1) << np.uint64(move) | f) & _FULL_MASK
            W = (W ^ f) & _FULL_MASK
        else:
            move = get_best_move(W, B, weights2, depth)
            if move == -1:
                if get_legal_moves(B, W) == 0:
                    break
                turn = 1
                continue
            f = get_flip(W, B, move)
            W = (W | np.uint64(1) << np.uint64(move) | f) & _FULL_MASK
            B = (B ^ f) & _FULL_MASK
        turn = -turn
    black_stones = count_bits(B)
    white_stones = count_bits(W)
    return black_stones - white_stones

def benchmark_weights():
    try:
        with open('weight_first.json', 'r') as f:
            weights1 = np.array(json.load(f), dtype=np.float64)
        with open('weight_second.json', 'r') as f:
            weights2 = np.array(json.load(f), dtype=np.float64)
    except:
        print('Failed to load weight files')
        return
    print(f'Weight_First: {len(weights1)} parameters')
    print(f'Weight_Second: {len(weights2)} parameters')
    print('=' * 50)
    wins1 = 0
    wins2 = 0
    draws = 0
    for game in range(2):
        print(f'\nGame {game + 1}:')
        if game == 0:
            result = play_game(weights1, weights2, depth=10)
            print(f'  Weight_First (Black) vs Weight_Second (White)')
        else:
            result = play_game(weights2, weights1, depth=10)
            print(f'  Weight_Second (Black) vs Weight_First (White)')
        if result > 0:
            print(f'  Black wins by {result} stones')
            if game == 0:
                wins1 += 1
            else:
                wins2 += 1
        elif result < 0:
            print(f'  White wins by {-result} stones')
            if game == 0:
                wins2 += 1
            else:
                wins1 += 1
        else:
            print('  Draw')
            draws += 1
    print('=' * 50)
    print(f'Final Results:')
    print(f'  Weight_First: {wins1} wins')
    print(f'  Weight_Second: {wins2} wins')
    print(f'  Draws: {draws}')
    if wins1 > wins2:
        print('  Weight_First wins the match!')
    elif wins2 > wins1:
        print('  Weight_Second wins the match!')
    else:
        print('  Match is a draw!')
if __name__ == '__main__':
    benchmark_weights()