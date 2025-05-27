import chess
import chess.pgn
import requests
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import re

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

class LinearAlgebraChessBot:
    def __init__(self):
        self.position_database = {}
        self.weight_matrix = None
        self.feature_weights = {
            'games_played': 0.3,
            'performance_rating': 0.4,
            'avg_rating': 0.2,
            'recency': 0.1
        }

    def extract_position_features(self, board: chess.Board) -> np.ndarray:
        features = np.zeros(8)
        material_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                           chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        white_material = sum(material_values[piece.piece_type]
                             for piece in board.piece_map().values()
                             if piece.color == chess.WHITE)
        black_material = sum(material_values[piece.piece_type]
                             for piece in board.piece_map().values()
                             if piece.color == chess.BLACK)
        features[0] = (white_material - black_material) / 39.0

        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        center_control = 0
        for square in center_squares:
            attackers_white = len(board.attackers(chess.WHITE, square))
            attackers_black = len(board.attackers(chess.BLACK, square))
            center_control += (attackers_white - attackers_black)
        features[1] = center_control / 8.0

        current_mobility = len(list(board.legal_moves))
        features[2] = current_mobility / 40.0

        white_developed = sum(1 for square, piece in board.piece_map().items()
                              if piece.color == chess.WHITE and
                              piece.piece_type in [chess.KNIGHT, chess.BISHOP] and
                              chess.square_rank(square) > 0)
        black_developed = sum(1 for square, piece in board.piece_map().items()
                              if piece.color == chess.BLACK and
                              piece.piece_type in [chess.KNIGHT, chess.BISHOP] and
                              chess.square_rank(square) < 7)
        features[3] = (white_developed - black_developed) / 4.0

        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)
        white_king_safety = -len(board.attackers(chess.BLACK, white_king_sq))
        black_king_safety = -len(board.attackers(chess.WHITE, black_king_sq))
        features[4] = (white_king_safety - black_king_safety) / 8.0

        features[5] = np.random.uniform(-0.1, 0.1)
        features[6] = np.random.uniform(-0.1, 0.1)
        features[7] = 0.1 if board.turn == chess.WHITE else -0.1

        return features

    def compute_move_weights(self, move_stats: List[MoveStatistics], current_year: int = 2024) -> np.ndarray:
        if not move_stats:
            return np.array([])

        weights = np.zeros((len(move_stats), 4))
        for i, stats in enumerate(move_stats):
            weights[i, 0] = np.log(max(1, stats.games_played))
            perf_diff = stats.performance_rating - stats.avg_rating
            weights[i, 1] = perf_diff / 200.0
            weights[i, 2] = (stats.avg_rating - 1500) / 500.0
            years_old = max(0, current_year - stats.last_played_year)
            weights[i, 3] = np.exp(-years_old / 5.0)
        return weights

    def calculate_move_probabilities(self, board: chess.Board, move_stats: List[MoveStatistics]) -> np.ndarray:
        if not move_stats:
            return np.array([])
        W = self.compute_move_weights(move_stats)
        w = np.array([
            self.feature_weights['games_played'],
            self.feature_weights['performance_rating'],
            self.feature_weights['avg_rating'],
            self.feature_weights['recency']
        ])
        position_features = self.extract_position_features(board)
        position_bias = np.sum(position_features) * 0.1
        raw_scores = W @ w + position_bias
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        probabilities = exp_scores / np.sum(exp_scores)
        return probabilities

    def select_move_linear_algebra(self, board: chess.Board, move_stats: List[MoveStatistics]) -> Optional[chess.Move]:
        if not move_stats:
            legal_moves = list(board.legal_moves)
            return np.random.choice(legal_moves) if legal_moves else None
        probabilities = self.calculate_move_probabilities(board, move_stats)
        if len(probabilities) == 0:
            legal_moves = list(board.legal_moves)
            return np.random.choice(legal_moves) if legal_moves else None
        selected_idx = np.random.choice(len(move_stats), p=probabilities)
        selected_stats = move_stats[selected_idx]
        try:
            return chess.Move.from_uci(selected_stats.move_uci)
        except:
            legal_moves = list(board.legal_moves)
            return np.random.choice(legal_moves) if legal_moves else None

class ChessTempoScraper:
    def __init__(self, headless=True):
        self.driver = None
        self.headless = headless
        self.setup_driver()

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
            board = chess.Board(fen)
            self.driver.get("https://www.chesstempo.com/opening-explorer")
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.ct-data-table")))
            time.sleep(2)
            stats = self.extract_move_stats_from_page()
            result = []
            for move_san, count in stats:
                try:
                    move = board.parse_san(move_san)
                    move_uci = move.uci()
                    result.append(MoveStatistics(move_san, move_uci, count, 0.0, 0.0, 0.0, 2200, 2250, 2025))
                except:
                    continue
            return result
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
                    else:
                        print("Illegal move.")
                except:
                    print("Invalid UCI format.")
            else:
                print("Bot is thinking...")
                move_stats = scraper.scrape_position_stats(board.fen())
                move = bot.select_move_linear_algebra(board, move_stats)
                if move and move in board.legal_moves:
                    print(f"Bot plays: {board.san(move)}")
                    board.push(move)
                else:
                    print("Bot failed to find a valid move.")
                    break
        print("\nGame Over! Result:", board.result())
    finally:
        scraper.close()

if __name__ == "__main__":
    play_against_bot()
