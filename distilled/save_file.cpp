
#include <algorithm>

#include "layers/generic.h"
#include "marian.h"
#include "models/decoder.h"
#include "models/encoder.h"
#include "models/model_factory.h"
#include "models/s2s.h"

using namespace marian;

int main() {
  auto graph = New<ExpressionGraph>();

  graph->setDevice({0, DeviceType::cpu});
  graph->reserveWorkspaceMB(128);

  const int dimEmb = 50;

  marian::Vocab vocab(New<Options>(), 0);
  vocab.load("/workspaces/bergamot/attentions/vocab.eten.spm");

  const auto words = vocab.encode("Hi dog");

  auto options =
  // Embeddings Options
      New<Options>("dimVocab", vocab.size(), 
                   "dimEmb", dimEmb,
                   "dropout-embeddings",false,
                   "inference", true,
                   "fixed", true,
                   "prefix", "src",
                   "enc-depth", 1 );
                   
  Embedding embedding(graph, options);

  const int wordSize = words.size();
  
  auto mask = graph->constant({wordSize, 1, 1 }, marian::inits::fromVector(std::vector<float>(wordSize, 1.0f)));
  
  auto embedding_src = embedding.apply(words, {1, static_cast<int>(words.size()), dimEmb});
  
  graph->forward();
  
  graph->save( "/workspaces/bergamot/bergamot-translator/distilled/test.npz");
  
  std::vector<float> values;
  
  embedding_src->val()->get(values);

  std::cout << values[0] << std::endl;

  return 0;
}
