// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "RunModelViewController.h"

#include <fstream>
#include <pthread.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/message_lite.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

#include "ios_image_load.h"

static bool initlized = false;
void InitModel();
std::vector<std::pair<std::string, float>> RunInference(const std::string& sent);

namespace {
class IfstreamInputStream : public ::google::protobuf::io::CopyingInputStream {
 public:
  explicit IfstreamInputStream(const std::string& file_name)
      : ifs_(file_name.c_str(), std::ios::in | std::ios::binary) {}
  ~IfstreamInputStream() { ifs_.close(); }

  int Read(void* buffer, int size) {
    if (!ifs_) {
      return -1;
    }
    ifs_.read(static_cast<char*>(buffer), size);
    return (int)ifs_.gcount();
  }

 private:
  std::ifstream ifs_;
};
}  // namespace

@interface RunModelViewController ()
@end

@implementation RunModelViewController {
}

- (void) viewDidLoad {
  if (!initlized) {
    initlized = true;
    InitModel();
  }
}

- (IBAction) inference:(id) sender {
  std::string sent([self.inputSentField.text UTF8String]);
  if (sent.empty()) {
    return;
  }
  auto predict = RunInference(sent);
  NSMutableString* inference_result = [NSMutableString string];
  for (const auto& tuple : predict) {
    const std::string& emoji = tuple.first;
    const float score = tuple.second;
    [inference_result appendFormat:@"%@: %.4f\n", [NSString stringWithUTF8String: emoji.c_str()], score];
  }
  self.urlContentTextView.text = inference_result;
}

- (IBAction) textEdited:(id)sender {
  NSDate* now = [NSDate date];
  if (self.lastFire == nil || [now timeIntervalSinceDate:self.lastFire] >= 0.2) {
    [self inference: nil];
    self.lastFire = now;
  }
}

@end

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
static void GetTopN(
    const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
                           Eigen::Aligned>& prediction,
    const int num_results, const float threshold,
    std::vector<std::pair<float, int> >* top_results) {
  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>,
      std::vector<std::pair<float, int> >,
      std::greater<std::pair<float, int> > > top_result_pq;

  const int count = (int)prediction.size();
  for (int i = 0; i < count; ++i) {
    const float value = prediction(i);

    // Only add it if it beats the threshold and has a chance at being in
    // the top N.
    if (value < threshold) {
      continue;
    }

    top_result_pq.push(std::pair<float, int>(value, i));

    // If at capacity, kick the smallest value out.
    if (top_result_pq.size() > num_results) {
      top_result_pq.pop();
    }
  }

  // Copy to output vector and reverse into descending order.
  while (!top_result_pq.empty()) {
    top_results->push_back(top_result_pq.top());
    top_result_pq.pop();
  }
  std::reverse(top_results->begin(), top_results->end());
}


bool PortableReadFileToProto(const std::string& file_name,
                             ::google::protobuf::MessageLite* proto) {
  ::google::protobuf::io::CopyingInputStreamAdaptor stream(
      new IfstreamInputStream(file_name));
  stream.SetOwnsCopyingStream(true);
  // TODO(jiayq): the following coded stream is for debugging purposes to allow
  // one to parse arbitrarily large messages for MessageLite. One most likely
  // doesn't want to put protobufs larger than 64MB on Android, so we should
  // eventually remove this and quit loud when a large protobuf is passed in.
  ::google::protobuf::io::CodedInputStream coded_stream(&stream);
  // Total bytes hard limit / warning limit are set to 1GB and 512MB
  // respectively. 
  coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
  return proto->ParseFromCodedStream(&coded_stream);
}

NSString* FilePathForResourceName(NSString* name, NSString* extension) {
  NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
  if (file_path == NULL) {
    LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "."
	       << [extension UTF8String] << "' in bundle.";
  }
  return file_path;
}

tensorflow::Session* session_pointer = nullptr;
std::unique_ptr<tensorflow::Session> session;

std::vector<std::string> label_strings = {
  "🎉", "🎈", "🙊", "🙄", "👑", "✨", "💞", "💕", "❤", "😏", "🔥", "😎", "💀",
  "😂", "😍", "😊", "😈", "❤️", "💔", "😅", "🌟", "😜", "😭", "💗", "😋", "🌹",
  "😩", "💦", "♂", "🙏", "☺", "💯", "😆", "➡️", "🙌", "💜", "✔", "💓", "💙",
  "😀", "👉", "😬", "👌", "😘", "♡", "🙃", "😁", "🙂", "👀", "💃", "💛", "👏",
  "👍", "😛", "💪", "💋", "😻", "😉", "😄", "😴", "💥", "💖", "😤", "🚨", "⚡",
  "😳", "🎶", "🗣", "👅", "😫", "✌", "💚", "🙈", "😇", "😒", "😌", "❗", "😢",
  "😕", "👊", "🌙", "👇", "😔", "❄", "💘", "✊", "💫", "😡", "♀", "🏆", "🌸",
  "★", "😱", "📷", "💰", "⚽", "🐐", "✅"
};

void InitModel() {
  tensorflow::SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(tensorflow::OptimizerOptions::L0);
  options.config.set_inter_op_parallelism_threads(1);
  options.config.set_intra_op_parallelism_threads(1);
  tensorflow::Status session_status = tensorflow::NewSession(options, &session_pointer);
  
  if (!session_status.ok()) {
    std::string status_string = session_status.ToString();
    NSLog(@"Session create failed - %s", status_string.c_str());
    return;
  }
  session.reset(session_pointer);
  LOG(INFO) << "Session created.";
  
  tensorflow::GraphDef tensorflow_graph;
  LOG(INFO) << "Graph created.";
  
  NSString* network_path = FilePathForResourceName(@"emoji_frozen", @"pb");  // tensorflow_inception_graph
  PortableReadFileToProto([network_path UTF8String], &tensorflow_graph);
  
  LOG(INFO) << "Creating session.";
  tensorflow::Status s = session->Create(tensorflow_graph);
  if (!s.ok()) {
    LOG(ERROR) << "Could not create TensorFlow Graph: " << s;
    return;
  }
}

// Generates feature sequence for a sentence.
tensorflow::Tensor TextToInputSequence(const std::string& sent) {
  // Everything here should be consistent with the original Python code (tokenize_dataset.ipynb).
  // Magic alphabet and label_strings are come from.
  tensorflow::Tensor text_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({1, 120}));
  auto tensor_mapped = text_tensor.tensor<tensorflow::int32, 2>();
  tensorflow::int32* data = tensor_mapped.data();

  // num_alphabet = 36  # (3+33)
  // num_cat = 99 # (1+98)
  // T_PAD = 0
  // T_OOV = 2
  const int T_START = 1;
  
  // Build alphabet.
  std::string alphabet = "### eotainsrlhuydmgwcpfbk.v'!,jx?zq_";
  std::map<char, int> aidx;
  for (int i = 0; i < alphabet.length(); ++i) {
    aidx[alphabet[i]] = i;
  }
  // Generate seq.
  std::vector<int> seq;
  seq.push_back(T_START);
  for (char ch : sent) {
    char lower_ch = tolower(ch);
    if (aidx.count(lower_ch) > 0) {
      seq.push_back(aidx[lower_ch]);
    }
  }
  // Trim and padding.
  const int MAX_LEN = 120;
  int seq_len = std::min(MAX_LEN, (int)seq.size());
  memset(data, 0, MAX_LEN * sizeof(int));
  memcpy(data + (MAX_LEN - seq_len), seq.data(), seq_len * sizeof(int));
  
  return text_tensor;
}

std::vector<std::pair<std::string, float>> RunInference(const std::string& sent) {
  std::vector<std::pair<std::string, float>> inference_result;
  // Extract feature.
  auto text_tensor = TextToInputSequence(sent);
  // Inference.
  std::string input_layer = "input_1";
  std::string output_layer = "dense_2/Softmax";
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::RunOptions options;
  tensorflow::RunMetadata metadata;
  tensorflow::Status run_status = session->Run(
      {{input_layer, text_tensor}}, {output_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    tensorflow::LogAllRegisteredKernels();
    return inference_result;
  }
  tensorflow::string status_string = run_status.ToString();
  LOG(INFO) << "Run status: " << status_string;
  
  // Collect outputs.
  tensorflow::Tensor* output = &outputs[0];
  const int kNumResults = 5;
  const float kThreshold = 0.005f;
  std::vector<std::pair<float, int>> top_results;
  GetTopN(output->flat<float>(), kNumResults, kThreshold, &top_results);
  
  std::stringstream ss;
  ss.precision(3);
  for (const auto& result : top_results) {
    const float confidence = result.first;
    const int index = result.second;
    const std::string& label = label_strings[index];
    ss << index << " " << confidence << "  " << label << "\n";
    inference_result.emplace_back(label, confidence);
  }
  LOG(INFO) << "Predictions: " << ss.str();
  return inference_result;
}
