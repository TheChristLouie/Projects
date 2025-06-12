import chess # Chess Library (Can store board, moves, etc.)
import chess.pgn
import requests
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from selenium import webdriver # Selenium allows automation of our web browser
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import re #String editing

@dataclass
class MoveStatistics:
    move_san: str
    move_uci: str
    games_played: int
    white_win_pct: float
    draw_pct: float
    black_win_pct: float
    avg_rating: int
    performance_rating: int
    last_played_year: int

#Used to move pieces on ChessTempo
def get_square_selector(square: str) -> str:
    return f'div[data-square-id="{square}"]'

class LinearAlgebraChessBot:
    #Bot is created
    def __init__(self):
        self.position_database = {}
        self.weight_matrix = None
        self.feature_weights = {
            'games_played': .4,
            'position_eval': .4,
            'avg_rating': .2
        }

    def extract_position_features(self, board: chess.Board) -> np.ndarray:
        features = np.zeros(5)
        # Material evaluation (1)
        material_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                           chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        white_material = sum(material_values[piece.piece_type]
                             for piece in board.piece_map().values()
                             if piece.color == chess.WHITE)
        black_material = sum(material_values[piece.piece_type]
                             for piece in board.piece_map().values()
                             if piece.color == chess.BLACK)
        features[0] = (white_material - black_material) / 39.0 *10

        # Center control (2)
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        center_control = 0
        for square in center_squares:
            attackers_white = len(board.attackers(chess.WHITE, square))
            attackers_black = len(board.attackers(chess.BLACK, square))
            center_control += (attackers_white - attackers_black)
        features[1] = center_control / 32.0 *.2

        # Mobility (3)
        current_turn = board.turn
        # Calculate mobility for both sides
        if current_turn == chess.WHITE:
            white_mobility = len(list(board.legal_moves))
            board.turn = chess.BLACK
            black_mobility = len(list(board.legal_moves))
            board.turn = chess.WHITE
        else:
            black_mobility = len(list(board.legal_moves))
            board.turn = chess.WHITE
            white_mobility = len(list(board.legal_moves))
            board.turn = chess.BLACK
        board.turn = current_turn
        total_mobility = white_mobility + black_mobility
        if total_mobility > 0:
            features[2] = (white_mobility - black_mobility) / total_mobility * .2
        else:
            features[2] = 0.0

        # Development (4)
        white_developed = 0
        black_developed = 0
        for square, piece in board.piece_map().items():
            if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                rank = chess.square_rank(square)
                if piece.color == chess.WHITE:
                    if rank > 0:
                        white_developed += 1
                else:
                    if rank < 7:
                        black_developed += 1
        
        features[3] = (white_developed - black_developed) / 4.0 *.2

        # Turn advantage
        features[4] = .1 if board.turn == chess.WHITE else -.1

        return features

    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate the current position from the perspective of the side to move.
        Positive values favor the side to move, negative values favor the opponent.
        """
        features = self.extract_position_features(board)
        # Calculate base position score
        position_score = np.sum(features)
        # Adjust perspective based on who's to move
        if board.turn == chess.BLACK:
            position_score = -position_score
        # Add special considerations
        if board.is_checkmate():
            return -1000.0  # Very bad for side to move
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0.0  # Draw
        
        return position_score

    def evaluate_move(self, board: chess.Board, move: chess.Move, move_stats: Optional[MoveStatistics] = None) -> float:
        """
        Evaluate a move by looking at the resulting position.
        """
        # Make the move temporarily
        board.push(move)
        # Evaluate the resulting position (from opponent's perspective, so negate)
        position_eval = -self.evaluate_position(board)
        # Undo the move
        board.pop()
        # Combine position evaluation with move statistics
        total_score = position_eval
        if move_stats:
            # Add bonus for frequently played moves
            frequency_bonus = min(move_stats.games_played / 10000, 0.2)
            # Add bonus for high-rated games
            rating_bonus = (move_stats.avg_rating - 2000) / 1000 * 0.1
            total_score += frequency_bonus + rating_bonus
        return total_score

    def calculate_move_probabilities(self, board: chess.Board, move_stats: List[MoveStatistics]) -> np.ndarray:
        if not move_stats:
            return np.array([])
        # Filter to only legal moves
        legal_move_ucis = {move.uci() for move in board.legal_moves}
        valid_move_stats = [stat for stat in move_stats if stat.move_uci in legal_move_ucis]
        if not valid_move_stats:
            return np.array([])
        # Evaluate each move
        scores = []
        for move_stat in valid_move_stats:
            try:
                move = chess.Move.from_uci(move_stat.move_uci)
                if move in board.legal_moves:
                    score = self.evaluate_move(board, move, move_stat)
                    scores.append(score)
                else:
                    scores.append(-1000.0)  # Illegal Move Fallback
            except Exception as e:
                print(f"Error evaluating move {move_stat.move_uci}: {e}")
                scores.append(-1000.0) # Illegal Move Fallback
        if not scores:
            return np.array([])
        scores = np.array(scores)
        # Convert to probabilities using softmax with temperature
        temperature = 0.5  # Lower temperature = more decisive
        exp_scores = np.exp(scores / temperature)
        probabilities = exp_scores / np.sum(exp_scores)
        return probabilities

    def select_move_linear_algebra(self, board: chess.Board, move_stats: List[MoveStatistics]) -> Optional[chess.Move]:
        if not move_stats:
            # If no move statistics, evaluate all legal moves
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None
            best_move = None
            best_score = float('-inf')
            for move in legal_moves:
                score = self.evaluate_move(board, move)
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_move
        
        # Filter to only legal moves
        legal_move_ucis = {move.uci() for move in board.legal_moves}
        valid_move_stats = [stat for stat in move_stats if stat.move_uci in legal_move_ucis]
        if not valid_move_stats:
            return self.select_move_linear_algebra(board, [])  # Fall back to pure evaluation
        probabilities = self.calculate_move_probabilities(board, valid_move_stats)
        if len(probabilities) == 0:
            return self.select_move_linear_algebra(board, [])  # Fall back to pure evaluation
        # Select move based on probabilities (with some randomness)
        if np.random.random() < 0.8:  # 80% of the time, pick the best move
            selected_idx = np.argmax(probabilities)
        else:  # 20% of the time, sample from distribution
            selected_idx = np.random.choice(len(valid_move_stats), p=probabilities)
        selected_stats = valid_move_stats[selected_idx]
        try:
            move = chess.Move.from_uci(selected_stats.move_uci)
            if move in board.legal_moves:
                return move
            else:
                return self.select_move_linear_algebra(board, [])
        except:
            return self.select_move_linear_algebra(board, [])

class ChessTempoScraper:
    def __init__(self, headless=True):
        self.driver = None
        self.headless = headless
        self.setup_driver()
        self.driver.get("https://chesstempo.com/game-database/")
        # Click the launch button to load the database page
        try:
            launch_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, "ct-gdb-launch-button"))
            )
            launch_button.click()
            print("Navigating to the Chess Game Database page...")
        except Exception as e:
            print(f"Error clicking launch button: {e}")

    def setup_driver(self):
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            print(f"Error setting up Chrome driver: {e}")
    
    def scrape_position_stats(self, fen: str) -> List[MoveStatistics]:
        try:
            board = chess.Board()  # Initialize board at the current position
            board.set_fen(fen)
            # Wait for the table to load
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.ct-data-table")))
            time.sleep(2)
            stats = self.extract_move_stats_from_page()
            result = []
            for move_san, count in stats:
                try:
                    #Remove move number prefix (e.g., "1.e4" → "e4")
                    move_san_cleaned = re.sub(r"^\d+\.*\s*", "", move_san).strip()
                    move = board.parse_san(move_san_cleaned)
                    move_uci = move.uci()
                    result.append(MoveStatistics(move_san_cleaned, move_uci, count, 0.0, 0.0, 0.0, 2200, 2250, 2025))
                except Exception as e:
                    print(f"Error parsing SAN '{move_san}': {e}")
                    continue
            return(result)
        except Exception as e:
            print(f"Scrape error: {e}")
            return []

    def extract_move_stats_from_page(self) -> List[Tuple[str, int]]:
        results = []
        try:
            rows = self.driver.find_elements(By.CSS_SELECTOR, "tr.ct-dt-row")
            for row in rows:
                try:
                    move_elem = row.find_element(By.CSS_SELECTOR, "span.ct-opex-cand-move")
                    count_elem = row.find_element(By.CSS_SELECTOR, "span.ct-opex-num-moves")
                    move_san = move_elem.text.strip()
                    games_played = int(count_elem.text.replace(',', '').strip())
                    results.append((move_san, games_played))
                except Exception:
                    continue
        except Exception as e:
            print("Failed to extract table:", e)
        return results

    def drag_and_drop_piece(self, move: chess.Move):
        try:
            source_square = move.uci()[:2]
            target_square = move.uci()[2:]
            source_selector = f'div[data-square-id="{source_square}"]'
            target_selector = f'div[data-square-id="{target_square}"]'
            print(f"Checking source square: {source_selector}")
            print(f"Checking target square: {target_selector}")
            source_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, source_selector))
            )
            target_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, target_selector))
            )
            print(f"Found source square {source_square} and target square {target_square}")
            actions = ActionChains(self.driver)
            actions.click_and_hold(source_element).move_to_element(target_element).release().perform()
            print(f"Bot moved piece from {source_square} to {target_square} on ChessTempo.")
            time.sleep(2)
        except Exception as e:
            print(f"ERROR: Failed to drag and drop piece '{move.uci()}': {e}")

    def close(self):
        if self.driver:
            self.driver.quit()

def play_against_bot():
    bot = LinearAlgebraChessBot()
    scraper = ChessTempoScraper(headless=False)
    board = chess.Board()
    try:
        user_is_white = input("Do you want to play as White? (y/n): ").strip().lower().startswith("y")
        while not board.is_game_over():
            print("\n" + "-" * 60)
            print(board)
            print(f"Turn: {'White' if board.turn else 'Black'}")
            if (board.turn == chess.WHITE and user_is_white) or (board.turn == chess.BLACK and not user_is_white):
                user_input = input("Your move (in UCI, e.g. e2e4): ").strip()
                try:
                    move = chess.Move.from_uci(user_input)
                    if move in board.legal_moves:
                        board.push(move)
                        scraper.drag_and_drop_piece(move)
                    else:
                        print("Illegal move.")
                except:
                    print("Invalid UCI format.")
            else:
                print("Bot is thinking...")
                print(f"Current position evaluation: {bot.evaluate_position(board):.3f}")
                move_stats = scraper.scrape_position_stats(board.fen())
                if not move_stats:
                    print("\nNo moves were scraped! Using pure position evaluation.")
                else:
                    print(f"\nMoves scraped successfully: {len(move_stats)} moves")
                probabilities = bot.calculate_move_probabilities(board, move_stats)
                if probabilities.size == 0:
                    print("\nNo probabilities generated! Using fallback evaluation.")
                else:
                    print("\nTop move evaluations:")
                    # Show top 3 moves with their evaluations
                    move_evals = []
                    legal_move_ucis = {move.uci() for move in board.legal_moves}
                    valid_move_stats = [stat for stat in move_stats if stat.move_uci in legal_move_ucis]
                    for i, (stats, prob) in enumerate(zip(valid_move_stats, probabilities)):
                        try:
                            move = chess.Move.from_uci(stats.move_uci)
                            eval_score = bot.evaluate_move(board, move, stats)
                            move_evals.append((stats.move_san, prob, eval_score))
                        except:
                            continue
                    # Sort by probability and show top 3
                    move_evals.sort(key=lambda x: x[1], reverse=True)
                    for i, (move_san, prob, eval_score) in enumerate(move_evals[:3]):
                        print(f"  {i+1}. {move_san}: probability={prob:.3f}, evaluation={eval_score:.3f}")
                move = bot.select_move_linear_algebra(board, move_stats)
                if move and move in board.legal_moves:
                    print(f"\nBot plays: {board.san(move)}")
                    scraper.drag_and_drop_piece(move)
                    board.push(move)
                else:
                    print("\nERROR: Bot failed to find a valid move. Check scraper or move selection logic.")
                    break        
        print("\nGame Over! Result:", board.result())
    finally:
        scraper.close()

if __name__ == "__main__":
    play_against_bot()