# Solver Strategy Experiments

Small local experiments on open puzzle environments. These are not final leaderboard
estimates; they are decision notes for which methods to keep, limit, or avoid.

## Setup

- One generated input set per environment.
- One train run per environment.
- Then multiple `SOLVER_STRATEGY` values on the same input/model.
- Short local budgets: `TRAIN_TIME_LIMIT=25`, `SOLVE_TIME_LIMIT=25`.

## Results

| Env | Instances | Strategy | Score | Solved | Decision |
| --- | ---: | --- | ---: | --- | --- |
| `toggle_lights` | 20 | `toggle_only` | 1.524977 | 20/20 | Keep as first solver for binary `TOGGLE`. |
| `toggle_lights` | 20 | `greedy` | 0.083333 | 1/20 | Avoid as primary. Too myopic. |
| `toggle_lights` | 20 | `beam` | 0.0 | 0/20 | Avoid for pure toggle when linear solver applies. |
| `toggle_lights` | 20 | `astar` | 0.0 | 0/20 | Avoid as primary for toggle. Learned V was not enough. |
| `toggle_lights` | 20 | `zero_astar` | 0.0 | 0/20 | Avoid; uninformed A* is too slow. |
| `toggle_lights` | 20 | `full` | 1.524977 | 20/20 | Keep; exact toggle solver dominates. |
| `game_15_2d` | 8 | `sliding_only` | 0.192308 | 1/8 | Limit; IDA* + Manhattan is too slow as primary. |
| `game_15_2d` | 8 | `greedy` | 0.0 | 0/8 | Avoid as primary. |
| `game_15_2d` | 8 | `beam` | 0.408811 | 3/8 | Keep as primary for sliding-like puzzles for now. |
| `game_15_2d` | 8 | `astar` | 0.178571 | 1/8 | Keep as fallback only. |
| `game_15_2d` | 8 | `zero_astar` | 0.0 | 0/8 | Avoid. |
| `game_15_2d` | 8 | `full` | 0.408811 | 3/8 | Keep after moving beam before expensive IDA*. |
| `game_15_2d` | 8 | `beam` + Manhattan/LC score | 0.0-0.192308 | 0-1/8 | Avoid as beam ranker; too brittle. |
| `game_15_2d` | 8 | `beam` + mismatch score | 0.506850 | 4/8 | Keep; smoother ranking preserves diversity. |
| `game_15_2d` | 8 | `full` direct sliding beam | 0.506850 | 4/8 | Keep as current default. |
| `cylinder_game` | 8 | `greedy` | 1.065476 | 5/8 | Keep as cheap early method candidate. |
| `cylinder_game` | 8 | `beam` | 0.75 | 3/8 | Limit; can waste time versus greedy/A*. |
| `cylinder_game` | 8 | `astar` | 1.25 | 5/8 | Keep. |
| `cylinder_game` | 8 | `zero_astar` | 0.0 | 0/8 | Avoid. |
| `cylinder_game` | 8 | `full` | 1.25 | 5/8 | Keep. |
| `game_15_2d` | 8 | reverse BFS table, 200k states | 0.192308 | 1/8 direct/near hit | Keep building block; depth 15 is too shallow alone. |
| `game_15_2d` | 8 | reverse BFS table + full | 0.408811 | 3/8 | Did not beat beam yet on this tiny table. |
| `game_15_2d` | 8 | old gym-based IDA* | 0.192308 | 1/8 | Avoid; env calls dominate. |
| `game_15_2d` | 8 | fast array IDA* | 1.205438 | 6/8 | Keep as primary for `EMPTY + SWAP`. |
| `game_15_2d` | 8 | full with fast array IDA* | 1.205438 | 6/8 | Current default. |
| `game_15_2d` | 8 | incremental Manhattan IDA* | 0.0 | 0/8 | Do not use by default; faster heuristic calls but current contour search fails to solve. |
| `game_15_2d` | 8 | full + retry reserve + dynamic safety margin | 1.392938 | 7/8 | Keep; better use of solve budget recovers one hard state at the same local limit. |
| `cylinder_game` | 8 | full + generic long beam before A* | 1.50 | 6/8 | Keep; beam is stronger than learned A* on rotational/color public puzzle. |
| `cylinder_game` | 8 | full + generic long beam, larger local budget | 1.75 | 7/8 | Confirms generic beam should get the main leftover budget. |

## Decisions

1. Exact structure-aware solvers should run before ML search.
   - Binary `TOGGLE` maps to GF(2) and is dramatically better than learned search.

2. Uninformed search should not be used except at tiny depths.
   - `zero_astar` repeatedly gets 0.0.

3. Greedy search is useful only as a cheap opportunistic pass.
   - It helps on `cylinder_game`, but fails on `game_15_2d` and mostly fails on `toggle_lights`.

4. IDA* with generic Manhattan is not strong enough as the primary sliding solver.
   - Manhattan + linear conflict is implemented, but still only solves 1/8 under the short local budget.
   - Keep it as a tiny fallback only. It must not consume the main budget.

5. Beam search is currently the best open generic fallback for sliding-like puzzles.
   - It should receive most of the per-instance budget when `EMPTY + SWAP` is detected.
   - Use smooth mismatch scoring for beam ordering. Manhattan/linear-conflict scoring reduced diversity and performed worse locally.

6. Learned `V(s)` is useful but not sufficient.
   - A* with `V` beats zero A*, but still loses to exact toggle and often to beam.

7. Compact state keys are now used in hot loops.
   - On a small 15-puzzle benchmark: JSON key ~7.14 us/state, compact key ~4.39 us/state.
   - The speedup is modest on array-like states but still useful and produces shorter keys.

8. Reverse BFS table works, but table depth matters.
   - Local 15-puzzle build with `BFS_TIME_LIMIT=8`, `BFS_MAX_STATES=200000`: about 53k states/sec, depth 15, 200k states.
   - This solved 1/8 directly or by shallow beam-to-table; deeper train-time tables are needed for real impact.
   - Pure `TOGGLE` puzzles skip BFS because GF(2) is exact and faster.

9. Fast native transitions are mandatory for sliding puzzles.
   - Old `ordered_children` on 15-puzzle: about 40k children/sec.
   - Fast list-swap plus compact key plus heuristic: about 114k children/sec in a direct microbenchmark.
   - More importantly, IDA* solved-rate jumped from 1/8 to 6/8 under the same short local budget.
   - The next optimization target is the fast heuristic itself, especially linear-conflict recomputation.

10. Incremental Manhattan is not ready.
   - Microbenchmark: Manhattan-only heuristic about 1.43 us/call, LC about 7.58 us/call.
   - Despite the cheaper heuristic, the incremental Manhattan IDA* variant solved 0/8 locally.
   - Keep LC fast IDA* as default; revisit incremental updates only after adding trace-level tests for IDA contours.

11. Solve-time allocation matters.
   - With short local solve limits, fixed per-instance fair sharing left early hard
     sliding instances unsolved. A small retry reserve plus dynamic safety margin
     improved `game_15_2d` from 6/8 to 7/8 on the same local set.

12. Generic/rotational puzzles should favor beam over learned A*.
   - On `cylinder_game`, long beam solved more instances than A*. The full cascade
     now gives non-sliding puzzles the main remaining per-instance budget for beam
     before falling back to A*.

## Next Tests

- Add a train-time reverse BFS table from solved state and measure lookup hit rate.
- Increase reverse BFS depth/state budget on `game_15_2d` and measure the crossover point where table+walkdown beats beam.
- Optimize fast sliding heuristic with incremental Manhattan and cheaper linear-conflict updates.
- Train a policy/action-ranker from reverse random walks and compare it against mismatch-only child ordering.
- Run staged solving: first pass all instances with cheap methods, then retry failures with larger budgets.
- Add multiprocessing carefully, making sure large lookup tables are shared by fork rather than copied per worker.
