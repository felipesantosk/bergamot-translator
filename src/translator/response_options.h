#ifndef SRC_BERGAMOT_RESPONSE_OPTIONS_H_
#define SRC_BERGAMOT_RESPONSE_OPTIONS_H_
#include <string>

namespace marian {
namespace bergamot {

enum ConcatStrategy {
  /// Target text is constructed faithful to the source-text  structure.
  FAITHFUL,

  /// Target text is concatenated by a space.
  SPACE
};

enum QualityScoreType {
  /// Provide a simple quality-score that comes with the machine-translation model
  /// itself.
  SIMPLE = 0,

  /// Provide a LogisticRegressor quality-score
  LR = 1,

  BEGIN_VALID_TYPE =  SIMPLE,
  END_VALID_TYPE =  LR
};

/// ResponseOptions dictate how to construct a Response for an input string of
/// text to be translated.
struct ResponseOptions {
  bool qualityScores{false};  ///< Include quality-scores or not.
  bool alignment{false};      ///< Include alignments or not.

  /// Whether to include sentenceMappings or not. Alignments require
  /// sentenceMappings and are available irrespective of this option if
  /// `alignment=true`.
  bool sentenceMappings{false};

  /// Threshold between `[0.0f, 1.0f]` to filter alignments into a sparse
  /// matrix. Higher value implies stronger filtering leading to provision of
  /// higher-confidence matches. `1.0f` gives argmax (not the full-dense
  /// matrix).
  float alignmentThreshold{0.2f};

  QualityScoreType qualityScoreType{QualityScoreType::SIMPLE};
  ConcatStrategy concatStrategy{ConcatStrategy::FAITHFUL};
};

}  // namespace bergamot
}  // namespace marian

#endif  //  SRC_BERGAMOT_RESPONSE_OPTIONS_H_
