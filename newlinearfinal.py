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
    """Statistics for a chess move from ChessTempo"""
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
    """
    Chess bot using linear algebra concepts for move selection.
    Mathematical Model:
    - Position State Vector: P ∈ R^n (position features)
    - Move Weight Matrix: W ∈ R^(m×k) (statistical weights for each move)
    - Probability Vector: π ∈ R^m (move selection probabilities)
    π = softmax(W · f(P))
    where f(P) is a feature extraction function
    """
    def __init__(self):
        self.position_database = {}  # FEN -> MoveStatistics[]
        self.weight_matrix = None
        self.feature_weights = {
            'games_played': 0.3,      # Popularity weight
            'performance_rating': 0.4, # Success weight  
            'avg_rating': 0.2,        # Player strength weight
            'recency': 0.1            # Time relevance weight
        }
    def extract_position_features(self, board: chess.Board) -> np.ndarray:
        """
        Extract numerical features from chess position for linear algebra operations.
        
        Returns feature vector P ∈ R^8:
        [material_balance, piece_activity, king_safety, center_control, 
         development, pawn_structure, space_advantage, tempo]
        """
        features = np.zeros(8)
        material_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                          chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        white_material = sum(material_values[piece.piece_type] 
                           for piece in board.piece_map().values() 
                           if piece.color == chess.WHITE)
        black_material = sum(material_values[piece.piece_type] 
                           for piece in board.piece_map().values() 
                           if piece.color == chess.BLACK)
        features[0] = (white_material - black_material) / 39.0  # Normalized material balance
        # Center control (squares e4, e5, d4, d5)
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        center_control = 0
        for square in center_squares:
            attackers_white = len(board.attackers(chess.WHITE, square))
            attackers_black = len(board.attackers(chess.BLACK, square))
            center_control += (attackers_white - attackers_black)
        features[1] = center_control / 8.0  # Normalized center control
        
        # Piece activity (mobility)
        current_mobility = len(list(board.legal_moves))
        # Simple approximation for opponent mobility
        features[2] = current_mobility / 40.0  # Normalized mobility
        
        # Development (pieces off back rank)
        white_developed = sum(1 for square, piece in board.piece_map().items()
                            if piece.color == chess.WHITE and 
                            piece.piece_type in [chess.KNIGHT, chess.BISHOP] and
                            chess.square_rank(square) > 0)
        black_developed = sum(1 for square, piece in board.piece_map().items()
                            if piece.color == chess.BLACK and 
                            piece.piece_type in [chess.KNIGHT, chess.BISHOP] and
                            chess.square_rank(square) < 7)
        
        features[3] = (white_developed - black_developed) / 4.0  # Normalized development
        
        # King safety (simplified)
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK) 
        white_king_safety = -len(board.attackers(chess.BLACK, white_king_sq))
        black_king_safety = -len(board.attackers(chess.WHITE, black_king_sq))
        features[4] = (white_king_safety - black_king_safety) / 8.0
        
        # Pawn structure (doubled, isolated pawns penalty)
        features[5] = np.random.uniform(-0.1, 0.1)  # Placeholder
        
        # Space advantage (squares controlled in opponent's half)
        features[6] = np.random.uniform(-0.1, 0.1)  # Placeholder
        
        # Tempo (who to move advantage)
        features[7] = 0.1 if board.turn == chess.WHITE else -0.1
        
        return features
    
    def compute_move_weights(self, move_stats: List[MoveStatistics], 
                           current_year: int = 2024) -> np.ndarray:
        """
        Compute weight matrix W ∈ R^(m×4) for m moves.
        Each row represents a move, columns are feature weights.
        """
        if not move_stats:
            return np.array([])
        
        weights = np.zeros((len(move_stats), 4))
        
        for i, stats in enumerate(move_stats):
            # Popularity weight (log scale for games played)
            weights[i, 0] = np.log(max(1, stats.games_played))
            
            # Performance weight (normalized rating difference)
            perf_diff = stats.performance_rating - stats.avg_rating
            weights[i, 1] = perf_diff / 200.0  # Normalize by ~200 rating points
            
            # Player strength weight (normalized average rating)
            weights[i, 2] = (stats.avg_rating - 1500) / 500.0  # Normalize around 1500±500
            
            # Recency weight (exponential decay)
            years_old = max(0, current_year - stats.last_played_year)
            weights[i, 3] = np.exp(-years_old / 5.0)  # 5-year half-life
        
        return weights
    
    def calculate_move_probabilities(self, board: chess.Board, 
                                   move_stats: List[MoveStatistics]) -> np.ndarray:
        """
        Calculate move selection probabilities using linear algebra.
        
        π = softmax(W @ w + b)
        where W is move weight matrix, w is feature weight vector, b is bias
        """
        if not move_stats:
            return np.array([])
        
        # Get weight matrix W ∈ R^(m×4)
        W = self.compute_move_weights(move_stats)
        
        # Feature weight vector w ∈ R^4
        w = np.array([
            self.feature_weights['games_played'],
            self.feature_weights['performance_rating'], 
            self.feature_weights['avg_rating'],
            self.feature_weights['recency']
        ])
        
        # Position features (incorporated as scalar bias)
        position_features = self.extract_position_features(board)
        position_bias = np.sum(position_features) * 0.1  # Scalar bias from position
        
        # Compute raw scores: s = W @ w + position_bias
        raw_scores = W @ w + position_bias  # Scalar bias applied to all moves
        
        # Apply softmax: π_i = exp(s_i) / Σ_j exp(s_j)  
        exp_scores = np.exp(raw_scores - np.max(raw_scores))  # Numerical stability
        probabilities = exp_scores / np.sum(exp_scores)
        
        return probabilities
    
    def select_move_linear_algebra(self, board: chess.Board, 
                                 move_stats: List[MoveStatistics]) -> Optional[chess.Move]:
        """Select move using linear algebra probability calculation"""
        if not move_stats:
            legal_moves = list(board.legal_moves)
            return np.random.choice(legal_moves) if legal_moves else None
        
        probabilities = self.calculate_move_probabilities(board, move_stats)
        
        if len(probabilities) == 0:
            legal_moves = list(board.legal_moves)
            return np.random.choice(legal_moves) if legal_moves else None
        
        # Select move based on calculated probabilities
        selected_idx = np.random.choice(len(move_stats), p=probabilities)
        selected_stats = move_stats[selected_idx]
        
        try:
            return chess.Move.from_uci(selected_stats.move_uci)
        except:
            legal_moves = list(board.legal_moves) 
            return np.random.choice(legal_moves) if legal_moves else None
    
    def display_linear_algebra_analysis(self, board: chess.Board, 
                                      move_stats: List[MoveStatistics]):
        """Display the linear algebra calculations for educational purposes"""
        if not move_stats:
            print("No move statistics available for analysis.")
            return
        
        print("\n" + "="*80)
        print("LINEAR ALGEBRA MOVE ANALYSIS")
        print("="*80)
        
        # Show position feature vector
        P = self.extract_position_features(board)
        print(f"\n1. POSITION FEATURE VECTOR P ∈ R^8:")
        feature_names = ['Material', 'Center', 'Mobility', 'Development', 
                        'King Safety', 'Pawns', 'Space', 'Tempo']
        for i, (name, value) in enumerate(zip(feature_names, P)):
            print(f"   P[{i}] = {value:+6.3f}  ({name})")
        
        # Show weight matrix
        W = self.compute_move_weights(move_stats)
        print(f"\n2. MOVE WEIGHT MATRIX W ∈ R^{W.shape[0]}×{W.shape[1]}:")
        print("   Columns: [Games, Performance, Rating, Recency]")
        for i, (stats, weights) in enumerate(zip(move_stats[:5], W[:5])):  # Show first 5
            print(f"   {stats.move_san:>4}: [{weights[0]:+6.2f}, {weights[1]:+6.2f}, "
                  f"{weights[2]:+6.2f}, {weights[3]:+6.2f}]")
        if len(move_stats) > 5:
            print(f"   ... ({len(move_stats)-5} more moves)")
        
        # Show feature weights
        w = np.array(list(self.feature_weights.values()))
        print(f"\n3. FEATURE WEIGHT VECTOR w ∈ R^4:")
        print(f"   w = [{w[0]:.1f}, {w[1]:.1f}, {w[2]:.1f}, {w[3]:.1f}]")
        print("   (Games, Performance, Rating, Recency)")
        
        # Show probability calculation
        probabilities = self.calculate_move_probabilities(board, move_stats)
        print(f"\n4. PROBABILITY VECTOR π = softmax(W @ w):")
        sorted_moves = sorted(zip(move_stats, probabilities), 
                            key=lambda x: x[1], reverse=True)
        
        for stats, prob in sorted_moves[:8]:  # Show top 8 moves
            print(f"   π({stats.move_san:>4}) = {prob:.3f}  "
                  f"({prob*100:5.1f}% | {stats.games_played} games)")
        
        print("\n" + "="*80)

class ChessTempoScraper:
    """Scraper for ChessTempo Opening Explorer data"""
    
    def __init__(self, headless=True):
        self.driver = None
        self.headless = headless
        self.setup_driver()
    
    def setup_driver(self):
        """Setup Selenium WebDriver"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            print(f"Error setting up Chrome driver: {e}")
            print("Please install ChromeDriver or use manual data entry mode.")
    
    def scrape_position_stats(self, fen: str) -> List[MoveStatistics]:
        """
        Scrape move statistics for a position from ChessTempo
        Note: This is a template - actual implementation would need to handle
        ChessTempo's specific HTML structure and any anti-bot measures
        """
        # For now, return mock data since web scraping needs proper setup
        return self.get_mock_data()
    
    def get_mock_data(self) -> List[MoveStatistics]:
        """Generate realistic mock data for testing"""
        mock_moves = [
            MoveStatistics("e4", "e2e4", 15420, 38.2, 32.1, 29.7, 2156, 2180, 2024),
            MoveStatistics("d4", "d2d4", 12890, 36.8, 35.2, 28.0, 2134, 2145, 2024), 
            MoveStatistics("Nf3", "g1f3", 8765, 35.1, 36.8, 28.1, 2098, 2112, 2023),
            MoveStatistics("c4", "c2c4", 4521, 37.9, 33.4, 28.7, 2187, 2201, 2024),
            MoveStatistics("g3", "g2g3", 1834, 34.2, 38.1, 27.7, 2076, 2089, 2023),
        ]
        return mock_moves
    
    def close(self):
        if self.driver:
            self.driver.quit()

def demonstrate_linear_algebra_bot():
    """Demonstrate the linear algebra chess bot"""
    print("CHESS BOT WITH LINEAR ALGEBRA ANALYSIS")
    print("="*60)
    
    bot = LinearAlgebraChessBot()
    scraper = ChessTempoScraper(headless=True)
    board = chess.Board()
    
    try:
        print(f"\nStarting Position:")
        print(board)
        
        # Get opening statistics
        print("\nFetching opening statistics...")
        move_stats = scraper.scrape_position_stats(board.fen())
        
        if move_stats:
            print(f"Found {len(move_stats)} candidate moves in database")
            
            # Display linear algebra analysis
            bot.display_linear_algebra_analysis(board, move_stats)
            
            # Select move using linear algebra
            selected_move = bot.select_move_linear_algebra(board, move_stats)
            
            if selected_move:
                move_san = board.san(selected_move)
                print(f"\nBOT SELECTED MOVE: {move_san}")
                board.push(selected_move)
                print(f"\nPosition after {move_san}:")
                print(board)
        else:
            print("No statistics available for this position")
    
    finally:
        scraper.close()

def play_full_game():
    """Play a full game using LinearAlgebraChessBot for both sides"""
    print("CHESS BOT FULL GAME DEMONSTRATION")
    print("=" * 60)

    bot = LinearAlgebraChessBot()
    scraper = ChessTempoScraper(headless=True)
    board = chess.Board()

    try:
        move_number = 1
        while not board.is_game_over():
            print("\n" + "-" * 60)
            print(f"Move {move_number} ({'White' if board.turn else 'Black'} to move):")
            print(board)

            fen = board.fen()
            move_stats = scraper.scrape_position_stats(fen)

            if not move_stats:
                print("No move statistics found. Terminating.")
                break

            bot.display_linear_algebra_analysis(board, move_stats)

            move = bot.select_move_linear_algebra(board, move_stats)
            if move is None:
                print("Bot failed to select a move. Terminating.")
                break

            move_san = board.san(move)
            print(f"\nBOT SELECTED MOVE: {move_san}")
            board.push(move)
            move_number += 1

        print("\n" + "=" * 60)
        print("GAME OVER")
        print(f"Result: {board.result()}")
        print("Final Position:")
        print(board)

    finally:
        scraper.close()

if __name__ == "__main__":
    play_full_game()