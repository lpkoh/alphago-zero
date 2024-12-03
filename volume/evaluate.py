# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evalation plays games between two neural nets."""

import os
from absl import app, flags
import dual_net
from strategies import MCTSPlayer
import utils

flags.DEFINE_string('eval_model', None, 'Path to the model for black player')
flags.DEFINE_integer('num_evaluation_games', 16, 'How many games to play')
flags.declare_key_flag('num_readouts')

FLAGS = flags.FLAGS

def play_match(black_model, white_model, games):
    """Plays matches between two neural nets.

    Args:
        black_model: Path to the model for black player
        white_model: Path to the model for white player
    """
    black_net = dual_net.DualNetwork(black_model)
    white_net = dual_net.DualNetwork(white_model)
    results = []

    black = MCTSPlayer(black_net, two_player_mode=True)
    white = MCTSPlayer(white_net, two_player_mode=True)

    for i in range(games):
        num_move = 0  # The move number of the current game

        for player in [black, white]:
            player.initialize_game()
            first_node = player.root.select_leaf()
            prob, val = player.network.run(first_node.position)
            first_node.incorporate_results(prob, val, first_node)

        while True:
            active = white if num_move % 2 else black
            inactive = black if num_move % 2 else white

            current_readouts = active.root.N
            while active.root.N < current_readouts + FLAGS.num_readouts:
                active.tree_search()

            # First, check the roots for hopeless games.
            if active.should_resign():  # Force resign
                active.set_result(-1 * active.root.position.to_play, was_resign=True)
                inactive.set_result(active.root.position.to_play, was_resign=True)

            if active.is_done():
                active.set_result(active.root.position.result(), was_resign=False)
                # Record result from black's perspective
                if active.result_string.startswith('B+'):
                    results.append('win')
                else:
                    results.append('lose')
                break

            move = active.pick_move()
            active.play_move(move)
            inactive.play_move(move)
            num_move += 1
    
    return results

def main(argv):
    eval_model_num = int(FLAGS.eval_model)
    results_dir = '/volume/outputs/comparisons'
    os.makedirs(results_dir, exist_ok=True)
    models_dir = '/volume/outputs/models'
    cur_model = f"{eval_model_num:06d}"
    cur_model_path = os.path.join(models_dir, cur_model)
    eval_results = []
    for i in range(1, eval_model_num):
        prev_model = f"{i:06d}"
        prev_model_path = os.path.join(models_dir, prev_model)
        results = play_match(prev_model, cur_model, FLAGS.num_evaluation_games)
        wins = results.count('win')
        eval_results.append((prev_model, wins))
    filepath = os.path.join(results_dir, f"eval_{cur_model}.csv")
    with open(filepath, 'w') as f:
        for prev_model, wins in eval_results:
            f.write(f"{prev_model},{wins}\n")

if __name__ == '__main__':
    flags.mark_flags_as_required([
        'eval_model'
    ])
    app.run(main)