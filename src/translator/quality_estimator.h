#ifndef SRC_BERGAMOT_QUALITY_ESTIMATOR_H_
#define SRC_BERGAMOT_QUALITY_ESTIMATOR_H_
#include <iostream>
#include <string>
#include <vector>

#include "annotation.h"
#include "definitions.h"
#include "intgemm/intgemm.h"
#include "logistic_regressor.h"

namespace marian {
namespace bergamot {
/// QualityEstimator (QE) is responsible for measuring the quality of a translation model.
/// It returns the probability of each translated term being a valid one.
/// It's worthwhile mentioning that a word is made of one or more byte pair encoding (BPE) tokens.

/// Currently, it only expects an AlignedMemory, which is given from a binary file.
/// It's expected from AlignedMemory the following structure:
/// - a header with the number of parameters dimensions
/// - a vector of standard deviations of features
/// - a vector of means of features
/// - a vector of coefficients
/// - a intercept value
///
/// Where each feature corresponds to one of the following:
/// - the mean BPE log Probabilities of a given word.
/// - the minimum BPE log Probabilities of a given word.
/// - the number of BPE tokens that a word is made of
/// - the overall mean considering all BPE tokens in a sentence

/// The current model implementation is a Logistic Model, so after the matrix multiply,
// there is a non-linear sigmoid transformation that converts the final scores into probabilities.
class QualityEstimator {
 public:
  /// WordsQualityEstimate contains the quality data of a given translated sentence.
  /// It includes the confidence (proxied by a probability) of each decoded word
  /// (higher probabilities imply better-translated words), the ByteRanges of each term,
  /// and the probability of the whole sentence, represented as the mean word scores.
  struct WordsQualityEstimate {
    std::vector<float> wordQualityScores;
    std::vector<ByteRange> wordByteRanges;
    float sentenceScore = 0.0;
  };

  /// Construct a QualityEstimator
  /// @param [in] logisticRegressor:
  explicit QualityEstimator(LogisticRegressor &&logisticRegressor);

  QualityEstimator(QualityEstimator &&other);

  static QualityEstimator fromAlignedMemory(const AlignedMemory &qualityEstimatorMemory);
  AlignedMemory toAlignedMemory() const;

  /// construct the struct WordsQualityEstimate
  /// @param [in] logProbs: the log probabilities given by an translation model
  /// @param [in] target: AnnotatedText target value
  /// @param [in] sentenceIdx: the id of a candidate sentence
  WordsQualityEstimate computeQualityScores(const std::vector<float> &logProbs, const AnnotatedText &target,
                                            const size_t sentenceIdx) const;

 private:
  /// Builds the words byte ranges and defines the WordFeature values
  /// @param [in] logProbs: the log probabilities given by an translation model
  /// @param [in] target: AnnotatedText target value
  /// @param [in] sentenceIdx: the id of a candidate sentence
  static std::pair<std::vector<ByteRange>, Matrix> mapBPEToWords(const std::vector<float> &logProbs,
                                                                 const AnnotatedText &target, const size_t sentenceIdx);

  LogisticRegressor logisticRegressor_;
};

}  // namespace bergamot
}  // namespace marian

#endif  // SRC_BERGAMOT_QUALITY_ESTIMATOR_H_
