#ifndef CTC_BEAM_SEARCH_DECODER_H_
#define CTC_BEAM_SEARCH_DECODER_H_

#include <string>
#include <vector>

#include "scorer.h"
#include "output.h"
#include "alphabet.h"

class DecoderState {
  int abs_time_step_;
  int space_id_;
  int blank_id_;
  int beam_size_;
  double cutoff_prob_;
  int cutoff_top_n_;

  Alphabet alphabet_;
  Scorer *ext_scorer_ = nullptr;

  std::vector<PathTrie*> prefixes_;
  std::unique_ptr<PathTrie> prefix_root_;

  bool has_scorer() const {
    return ext_scorer_ != nullptr;
  }

  bool has_word_level_scorer() const {
    return has_scorer() && !ext_scorer_->is_character_based();
  }

  bool has_char_level_scorer() const {
    return has_scorer() && !ext_scorer_->is_character_based();
  }

public:
  DecoderState() = default;
  ~DecoderState() = default;

  DecoderState(const DecoderState&) = delete;
  DecoderState& operator=(const DecoderState&) = delete;

  /* Initialize CTC beam search decoder
   *
   * Parameters:
   *     alphabet: The alphabet.
   *     beam_size: The width of beam search.
   *     cutoff_prob: Cutoff probability for pruning.
   *     cutoff_top_n: Cutoff number for pruning.
   *     ext_scorer: External scorer to evaluate a prefix, which consists of
   *                 n-gram language model scoring and word insertion term.
   *                 Default null, decoding the input sample without scorer.
   * Return:
   *     Zero on success, non-zero on failure.
  */
  int init(const Alphabet &alphabet,
           int beam_size,
           double cutoff_prob,
           int cutoff_top_n,
           Scorer* ext_scorer);

  /* Send data to the decoder
   *
   * Parameters:
   *     probs: 2-D vector where each element is a vector of probabilities
   *               over alphabet of one time step.
   *     time_dim: Number of timesteps.
   *     class_dim: Alphabet length (plus 1 for space character).
  */
  void next(const double *probs,
            int time_dim,
            int class_dim);

  /* Get transcription for the data you sent via next()
   *
   * Return:
   *     A vector where each element is a pair of score and decoding result,
   *     in descending order.
  */
  std::vector<Output> decode();
};

/* CTC Beam Search Decoder
 * Parameters:
 *     probs: 2-D vector where each element is a vector of probabilities
 *            over alphabet of one time step.
 *     time_dim: Number of timesteps.
 *     class_dim: Alphabet length (plus 1 for space character).
 *     alphabet: The alphabet.
 *     beam_size: The width of beam search.
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 * Return:
 *     A vector where each element is a pair of score and decoding result,
 *     in descending order.
*/

std::vector<Output> ctc_beam_search_decoder(
    const double* probs,
    int time_dim,
    int class_dim,
    const Alphabet &alphabet,
    size_t beam_size,
    double cutoff_prob,
    size_t cutoff_top_n,
    Scorer *ext_scorer);

/* CTC Beam Search Decoder for batch data
 * Parameters:
 *     probs: 3-D vector where each element is a 2-D vector that can be used
 *                by ctc_beam_search_decoder().
 *     alphabet: The alphabet.
 *     beam_size: The width of beam search.
 *     num_processes: Number of threads for beam search.
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 * Return:
 *     A 2-D vector where each element is a vector of beam search decoding
 *     result for one audio sample.
*/
std::vector<std::vector<Output>>
ctc_beam_search_decoder_batch(
    const double* probs,
    int batch_size,
    int time_dim,
    int class_dim,
    const int* seq_lengths,
    int seq_lengths_size,
    const Alphabet &alphabet,
    size_t beam_size,
    size_t num_processes,
    double cutoff_prob,
    size_t cutoff_top_n,
    Scorer *ext_scorer);

#endif  // CTC_BEAM_SEARCH_DECODER_H_
