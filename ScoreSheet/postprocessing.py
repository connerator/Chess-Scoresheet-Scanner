
import chess
import chess.pgn

def error_correct(move_candidates):
    moves = []
    board = chess.Board()
    for candidate_list in move_candidates:
        color = 'white' if board.turn == chess.WHITE else 'black'
        for _, san in candidate_list:
            try:
                move = board.parse_san(san)
                board.push(move)
                moves.append(san)
                break
            except:
                continue
        else:
            #raise Exception("No legal move detected:", candidate_list)
            print(moves)
            #print(candidate_list)
            manual_san = input(f"What move did {color} make on move {board.fullmove_number}?\n{board}\n\n")
            try:
                move = board.push_san(manual_san)
                moves.append(manual_san)
            except:
                raise Exception("Invalid move. Either you entered an illegal move, or a move was incorrectly recognized in an earlier move.")

    

    return moves

def make_pgn(moves):
    pgn = ""
    for i in range(0, len(moves), 2):
        pgn += f" {i//2 + 1}. "
        white_move = moves[i]
        pgn += white_move + " "
        if i+1 < len(moves):
            black_move = moves[i+1]
            pgn += black_move
    return pgn


def make_lichess_url(moves):
    return 'https://lichess.org/analysis/pgn/' + '_'.join(moves)
