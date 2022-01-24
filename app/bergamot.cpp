#include "translator/byte_array_util.h"
#include "translator/parser.h"
#include "translator/response.h"
#include "translator/response_options.h"
#include "translator/service.h"
#include "translator/utils.h"

int main(int argc, char *argv[]) {
  using namespace marian::bergamot;
  ConfigParser<AsyncService> configParser("Bergamot CLI", /*multiOpMode=*/false);
  configParser.parseArgs(argc, argv);
  auto &config = configParser.getConfig();

  AsyncService service(config.serviceConfig);

  // Construct a model.
  auto options = parseOptionsFromFilePath(config.modelConfigPaths.front());

  MemoryBundle memoryBundle;
  std::shared_ptr<TranslationModel> model = service.createCompatibleModel(options, std::move(memoryBundle));

  ResponseOptions responseOptions;
  responseOptions.qualityScores = true;

  std::string input = readFromStdin();

  // Create a barrier using future/promise.
  std::promise<Response> promise;
  std::future<Response> future = promise.get_future();
  auto callback = [&promise](Response &&response) {
    // Fulfill promise.
    promise.set_value(std::move(response));
  };

  service.translate(model, std::move(input), callback, responseOptions);

  // Wait until promise sets the response.
  Response response = future.get();

  // Print (only) translated text.
  // std::cout << response.target.text;

  std::cout << "[src Sentence]:\n" << response.source.text << "\n";
  std::cout << "[tgt Sentence]:\n" << response.target.text << "\n";
  std::cout << "[bpe Tokens]:";
  std::cout << std::fixed << std::setprecision(5);

  for (size_t s = 0; s < response.target.numSentences(); ++s) {
    const auto &sentenceQualityEstimate = response.qualityScores[s];
    for (size_t i = 0; i < sentenceQualityEstimate.logProbs.size(); ++i) {
      if (!sentenceQualityEstimate.bpeTokens[i].empty()) {
        std::cout << sentenceQualityEstimate.bpeTokens[i] << "(" << sentenceQualityEstimate.logProbs[i] << ")";
      }
    }
  }
  
  std::cout << "[Output]:" << "\n";
 
   for (size_t s = 0; s < response.target.numSentences(); ++s) {
     const auto &sentenceQualityEstimate = response.qualityScores[s];
    
    std::cout << s << " |||";
    for (size_t i = 0; i < sentenceQualityEstimate.words.size(); ++i) {
        std::cout << " " << ( *model->getVocabs().target() )[sentenceQualityEstimate.words[i] ];
    }
    
    std::cout << " ||| WordScores=";
 
    for (size_t i = 0; i < sentenceQualityEstimate.logProbs.size(); ++i) {
        std::cout << " " << sentenceQualityEstimate.logProbs[i];
     }
    
    std::cout << "\n";
   }

  
  std::cout << "[WordIndexes]:" << "\n";
  for (size_t s = 0; s < response.target.numSentences(); ++s) {
    const auto &sentenceQualityEstimate = response.qualityScores[s];
    for (size_t i = 0; i < sentenceQualityEstimate.words.size(); ++i) {
        std::cout << sentenceQualityEstimate.words[i].toWordIndex() << " ";
       }
   std::cout << "\n";
  }

  return 0;
}
