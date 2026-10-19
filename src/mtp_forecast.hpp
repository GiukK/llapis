// ===========================================================================
// mtp_forecast.hpp -- the MTP-state expert forecaster for the residency policy (claude dump findings 216j).
// ===========================================================================
// A small MLP trained offline on the bench (gemma-moe-lab/tools/p0b_window_bench.py, exported by
// tools/mtp_forecast_export.py) predicts for every (layer, expert) the fraction of the next H verify steps of the
// turn that use it. Inputs: the drafter's h_nextn for the drafts the last verify used (3 x 2816; what
// common/speculative.cpp holds on the host after each draft; zeros on the first verify of a turn, as in training)
// and the logit of a per-step usage EMA (half-life 4 steps, decode steps only). The policy ranks and prices its
// promotions on f = H * p instead of its own rate EMA (moert_policy_set_external_forecast).
// Threads: set_draft on the server thread while drafting; snapshot at the pre-decode hook (server thread, after the
// previous policy hook joined); add_counts / close_* / forecast on the policy hook. Pure CPU, no ggml / CUDA.
#pragma once
#include <cstdint>

namespace mtp_forecast {

bool load(const char * path, int n_layers, int n_expert);
bool active();
int  horizon();

// the drafter's h_nextn of draft i (0..2) of the current drafting round (n floats; a new round starts at i = 0)
void set_draft(int i, const float * h, int n);
// before a verify: the drafts it uses become the forecaster's input (turn_start: the first verify after a prompt
// -> zeros, as in training)
void snapshot(bool turn_start);
void add_counts(const int32_t * counts);
void close_step();     // a verify step ended: its counts become one step of usage (EMA)
void close_prompt();   // a prompt ended: its counts are dropped (training saw decode steps only)
double forecast(float * out);   // f = H * p [n_layers * n_expert]; returns ms

}  // namespace mtp_forecast
