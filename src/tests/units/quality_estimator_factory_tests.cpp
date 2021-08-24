#include "catch.hpp"
#include "translator/logistic_regressor_qe.h"
#include "translator/quality_estimator_factory.h"
#include "translator/unsupervised_qe.h"

using namespace marian::bergamot;

SCENARIO("Quality Estimator Factory test", "[QualityEstimatorFactory]") {
  WHEN("It's call Make with a empty AlignedMemory") {
    AlignedMemory emptyMemory;
    const auto model = QualityEstimatorFactory::Make(emptyMemory);

    THEN("It's created a UnsupervisedQE") {
      CHECK(dynamic_cast<const UnsupervisedQE*>(model.get()) != nullptr);
    }
  }
  WHEN("It's call Make with a LR AlignedMemory") {
    std::vector<float> coefficients = {0.99000001, 0.899999976, -0.200000003, 0.5};
    const float intercept = {-0.300000012};

    LogisticRegressorQE::Scale scale;
    scale.stds = {0.200000003, 0.300000012, 2.5, 0.100000001};
    scale.means = {-0.100000001, -0.769999981, 5, -0.5};

    LogisticRegressorQE logisticRegressor(std::move(scale), std::move(coefficients), intercept);

    const auto model = QualityEstimatorFactory::Make(logisticRegressor.toAlignedMemory());

    THEN("It's created a LogisticRegressor") { CHECK(dynamic_cast<const LogisticRegressorQE*>(model.get()) != nullptr); }
  }
}
