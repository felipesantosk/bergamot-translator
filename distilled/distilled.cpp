
#include <algorithm>

#include "layers/generic.h"
#include "marian.h"
#include "models/decoder.h"
#include "models/encoder.h"
#include "models/model_factory.h"
#include "models/s2s.h"

using namespace marian;

Expr createTextExprs(Ptr<ExpressionGraph>& graph, 
                     Ptr<Options>& options,
                     const Words& words,
                     const int dimEmb) {
  Embedding embedding(graph, options);

  const int wordSize = words.size();
  
  auto mask = graph->constant({wordSize, 1, 1 }, marian::inits::fromVector(std::vector<float>(wordSize, 1.0f)));
  
  auto embedding_src = embedding.apply(words, {1, static_cast<int>(words.size()), dimEmb});
  
  marian::EncoderS2S encoderS2S(graph, options);

  auto encoded_text_src = encoderS2S.applyEncoderRNN(graph, embedding_src, mask, "bidirectional");
  
  auto softMaxMask = graph->constant({1, wordSize, 1 }, marian::inits::fromVector(std::vector<float>(wordSize, 1.0f)));

  auto encState = New<EncoderState>(encoded_text_src, softMaxMask, nullptr);

  options->set("dimState", wordSize);

  std::vector<float> vState( wordSize );
  std::generate(vState.begin(), vState.end(), []() {
    static int n = -32;
    return n++ / 64.f;
  });
  
  auto attention = New<rnn::Attention>(graph, options, encState);
  rnn::State state({graph->constant({1, 1, wordSize }, inits::fromVector(vState)), nullptr});
  
  auto aligned = attention->apply(state);

  return aligned;
}

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
                   
  // RNN Options
  options->set<std::string>("enc-cell", "gru");
  options->set<int>("dim-rnn", dimEmb);
  options->set<float>("dropout-rnn", 0.1f);
  options->set<bool>("layer-normalization", false);
  options->set<bool>("skip", false);
  options->set<int>("enc-cell-depth", 1);
  
  auto aligned = createTextExprs(graph, options, words, dimEmb );
  
  graph->forward();
  
  std::vector<float> preds;
  
  aligned->val()->get(preds);

  std::cout << preds[0] << std::endl;

  return 0;
}
