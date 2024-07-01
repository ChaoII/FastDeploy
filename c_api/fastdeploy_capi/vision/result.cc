// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fastdeploy_capi/vision/result.h"

#include "fastdeploy/utils/utils.h"
#include "fastdeploy_capi/internal/types_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

// Classification Results

FD_C_ClassifyResultWrapper* FD_C_CreateClassifyResultWrapper() {
  FD_C_ClassifyResultWrapper* fd_c_classify_result_wrapper =
      new FD_C_ClassifyResultWrapper();
  fd_c_classify_result_wrapper->classify_result =
      std::unique_ptr<fastdeploy::vision::ClassifyResult>(
          new fastdeploy::vision::ClassifyResult());
  return fd_c_classify_result_wrapper;
}

void FD_C_DestroyClassifyResultWrapper(
    __fd_take FD_C_ClassifyResultWrapper* fd_c_classify_result_wrapper) {
  delete fd_c_classify_result_wrapper;
}

FD_C_ClassifyResult* FD_C_CreateClassifyResult() {
  FD_C_ClassifyResult* fd_c_classify_result = new FD_C_ClassifyResult();
  return fd_c_classify_result;
}

void FD_C_DestroyClassifyResultInner(
    __fd_take FD_C_ClassifyResult* fd_c_classify_result) {
  if (fd_c_classify_result == nullptr)
    return;
  // delete label_ids
  delete[] fd_c_classify_result->label_ids.data;
  // delete scores
  delete[] fd_c_classify_result->scores.data;
}

void FD_C_DestroyClassifyResult(
    __fd_take FD_C_ClassifyResult* fd_c_classify_result) {
  if (fd_c_classify_result == nullptr)
    return;
  FD_C_DestroyClassifyResultInner(fd_c_classify_result);
  delete fd_c_classify_result;
}

FD_C_OneDimClassifyResult* FD_C_CreateOneDimClassifyResult() {
  FD_C_OneDimClassifyResult* fd_c_one_dim_classify_result =
      new FD_C_OneDimClassifyResult();
  return fd_c_one_dim_classify_result;
}

void FD_C_DestroyOneDimClassifyResultInner(
    __fd_take FD_C_OneDimClassifyResult* fd_c_one_dim_classify_result) {
  for (int i = 0; i < fd_c_one_dim_classify_result->size; i++) {
    FD_C_DestroyClassifyResultInner(fd_c_one_dim_classify_result->data + i);
  }
  delete[] fd_c_one_dim_classify_result->data;
}

void FD_C_DestroyOneDimClassifyResult(
    __fd_take FD_C_OneDimClassifyResult* fd_c_one_dim_classify_result) {
  FD_C_DestroyOneDimClassifyResultInner(fd_c_one_dim_classify_result);
  delete fd_c_one_dim_classify_result;
}

void FD_C_ClassifyResultWrapperToCResult(
    __fd_keep FD_C_ClassifyResultWrapper* fd_c_classify_result_wrapper,
    __fd_keep FD_C_ClassifyResult* fd_c_classify_result_data) {
  auto& classify_result = CHECK_AND_CONVERT_FD_TYPE(
      ClassifyResultWrapper, fd_c_classify_result_wrapper);
  // copy label_ids
  fd_c_classify_result_data->label_ids.size = classify_result->label_ids.size();
  fd_c_classify_result_data->label_ids.data =
      new int32_t[fd_c_classify_result_data->label_ids.size];
  memcpy(fd_c_classify_result_data->label_ids.data,
         classify_result->label_ids.data(),
         sizeof(int32_t) * fd_c_classify_result_data->label_ids.size);
  // copy scores
  fd_c_classify_result_data->scores.size = classify_result->scores.size();
  fd_c_classify_result_data->scores.data =
      new float[fd_c_classify_result_data->scores.size];
  memcpy(fd_c_classify_result_data->scores.data, classify_result->scores.data(),
         sizeof(float) * fd_c_classify_result_data->scores.size);
  fd_c_classify_result_data->type =
      static_cast<FD_C_ResultType>(classify_result->type);
}

FD_C_ClassifyResultWrapper* FD_C_CreateClassifyResultWrapperFromCResult(
    __fd_keep FD_C_ClassifyResult* fd_c_classify_result) {
  FD_C_ClassifyResultWrapper* fd_c_classify_result_wrapper =
      FD_C_CreateClassifyResultWrapper();
  auto& classify_result = CHECK_AND_CONVERT_FD_TYPE(
      ClassifyResultWrapper, fd_c_classify_result_wrapper);
  // copy label_ids
  classify_result->label_ids.resize(fd_c_classify_result->label_ids.size);
  memcpy(classify_result->label_ids.data(),
         fd_c_classify_result->label_ids.data,
         sizeof(int32_t) * fd_c_classify_result->label_ids.size);
  // copy scores
  classify_result->scores.resize(fd_c_classify_result->scores.size);
  memcpy(classify_result->scores.data(), fd_c_classify_result->scores.data,
         sizeof(int32_t) * fd_c_classify_result->scores.size);
  classify_result->type =
      static_cast<fastdeploy::vision::ResultType>(fd_c_classify_result->type);
  return fd_c_classify_result_wrapper;
}

void FD_C_ClassifyResultStr(FD_C_ClassifyResult* fd_c_classify_result,
                            char* str_buffer) {
  FD_C_ClassifyResultWrapper* fd_c_classify_result_wrapper =
      FD_C_CreateClassifyResultWrapperFromCResult(fd_c_classify_result);
  auto& classify_result = CHECK_AND_CONVERT_FD_TYPE(
      ClassifyResultWrapper, fd_c_classify_result_wrapper);
  std::string information = classify_result->Str();
  std::strcpy(str_buffer, information.c_str());
  FD_C_DestroyClassifyResultWrapper(fd_c_classify_result_wrapper);
}

// Detection Results
FD_C_DetectionResultWrapper* FD_C_CreateDetectionResultWrapper() {
  FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper =
      new FD_C_DetectionResultWrapper();
  fd_c_detection_result_wrapper->detection_result =
      std::unique_ptr<fastdeploy::vision::DetectionResult>(
          new fastdeploy::vision::DetectionResult());
  return fd_c_detection_result_wrapper;
}

void FD_C_DestroyDetectionResultWrapper(
    __fd_take FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper) {
  delete fd_c_detection_result_wrapper;
}

FD_C_DetectionResult* FD_C_CreateDetectionResult() {
  FD_C_DetectionResult* fd_c_detection_result = new FD_C_DetectionResult();
  return fd_c_detection_result;
}

FD_C_OneDimDetectionResult* FD_C_CreateOneDimDetectionResult() {
  FD_C_OneDimDetectionResult* fd_c_one_dim_detection_result =
      new FD_C_OneDimDetectionResult();
  return fd_c_one_dim_detection_result;
}

void FD_C_DestroyDetectionResultInner(
    __fd_take FD_C_DetectionResult* fd_c_detection_result) {
  if (fd_c_detection_result == nullptr)
    return;
  // delete boxes
  for (size_t i = 0; i < fd_c_detection_result->boxes.size; i++) {
    delete[] fd_c_detection_result->boxes.data[i].data;
  }
  delete[] fd_c_detection_result->boxes.data;
  //delete rotated_boxes
  for (size_t i = 0; i < fd_c_detection_result->rotated_boxes.size; i++) {
    delete[] fd_c_detection_result->rotated_boxes.data[i].data;
  }
  delete[] fd_c_detection_result->rotated_boxes.data;
  // delete scores
  delete[] fd_c_detection_result->scores.data;
  // delete label_ids
  delete[] fd_c_detection_result->label_ids.data;
  // delete masks
  for (size_t i = 0; i < fd_c_detection_result->masks.size; i++) {
    delete[] fd_c_detection_result->masks.data[i].data.data;
    delete[] fd_c_detection_result->masks.data[i].shape.data;
  }
  delete[] fd_c_detection_result->masks.data;
}

void FD_C_DestroyDetectionResult(
    __fd_take FD_C_DetectionResult* fd_c_detection_result) {
  FD_C_DestroyDetectionResultInner(fd_c_detection_result);
}

void FD_C_DestroyOneDimDetectionResultInner(
    FD_C_OneDimDetectionResult* fd_c_one_dim_delete_result) {
  if (fd_c_one_dim_delete_result == nullptr) {
    return;
  }
  for (int i = 0; i < fd_c_one_dim_delete_result->size; i++) {
    FD_C_DestroyDetectionResultInner(fd_c_one_dim_delete_result->data + i);
  }
  delete[] fd_c_one_dim_delete_result->data;
}

void FD_C_DestroyOneDimDetectionResult(
    FD_C_OneDimDetectionResult* fd_c_one_dim_delete_result) {
  FD_C_DestroyOneDimDetectionResultInner(fd_c_one_dim_delete_result);
  delete fd_c_one_dim_delete_result;
}

void FD_C_DetectionResultWrapperToCResult(
    __fd_keep FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper,
    __fd_keep FD_C_DetectionResult* fd_c_detection_result) {
  auto& detection_result = CHECK_AND_CONVERT_FD_TYPE(
      DetectionResultWrapper, fd_c_detection_result_wrapper);
  // copy boxes
  const int boxes_coordinate_dim = 4;
  fd_c_detection_result->boxes.size = detection_result->boxes.size();
  fd_c_detection_result->boxes.data =
      new FD_C_OneDimArrayFloat[fd_c_detection_result->boxes.size];
  for (size_t i = 0; i < detection_result->boxes.size(); i++) {
    fd_c_detection_result->boxes.data[i].size = boxes_coordinate_dim;
    fd_c_detection_result->boxes.data[i].data = new float[boxes_coordinate_dim];
    for (size_t j = 0; j < boxes_coordinate_dim; j++) {
      fd_c_detection_result->boxes.data[i].data[j] =
          detection_result->boxes[i][j];
    }
  }
  // copy rotated_boxes
  const int rotated_boxes_coordinate_dim = 8;
  fd_c_detection_result->rotated_boxes.size =
      detection_result->rotated_boxes.size();
  fd_c_detection_result->rotated_boxes.data =
      new FD_C_OneDimArrayFloat[fd_c_detection_result->rotated_boxes.size];
  for (size_t i = 0; i < detection_result->rotated_boxes.size(); i++) {
    fd_c_detection_result->rotated_boxes.data[i].size =
        rotated_boxes_coordinate_dim;
    fd_c_detection_result->rotated_boxes.data[i].data =
        new float[rotated_boxes_coordinate_dim];
    for (size_t j = 0; j < rotated_boxes_coordinate_dim; j++) {
      fd_c_detection_result->rotated_boxes.data[i].data[j] =
          detection_result->rotated_boxes[i][j];
    }
  }
  // copy scores
  fd_c_detection_result->scores.size = detection_result->scores.size();
  fd_c_detection_result->scores.data =
      new float[fd_c_detection_result->scores.size];
  memcpy(fd_c_detection_result->scores.data, detection_result->scores.data(),
         sizeof(float) * fd_c_detection_result->scores.size);
  // copy label_ids
  fd_c_detection_result->label_ids.size = detection_result->label_ids.size();
  fd_c_detection_result->label_ids.data =
      new int32_t[fd_c_detection_result->label_ids.size];
  memcpy(fd_c_detection_result->label_ids.data,
         detection_result->label_ids.data(),
         sizeof(int32_t) * fd_c_detection_result->label_ids.size);
  // copy masks
  fd_c_detection_result->masks.size = detection_result->masks.size();
  fd_c_detection_result->masks.data =
      new FD_C_Mask[fd_c_detection_result->masks.size];
  for (size_t i = 0; i < detection_result->masks.size(); i++) {
    // copy data in mask
    fd_c_detection_result->masks.data[i].data.size =
        detection_result->masks[i].data.size();
    fd_c_detection_result->masks.data[i].data.data =
        new uint8_t[detection_result->masks[i].data.size()];
    memcpy(fd_c_detection_result->masks.data[i].data.data,
           detection_result->masks[i].data.data(),
           sizeof(uint8_t) * detection_result->masks[i].data.size());
    // copy shape in mask
    fd_c_detection_result->masks.data[i].shape.size =
        detection_result->masks[i].shape.size();
    fd_c_detection_result->masks.data[i].shape.data =
        new int64_t[detection_result->masks[i].shape.size()];
    memcpy(fd_c_detection_result->masks.data[i].shape.data,
           detection_result->masks[i].shape.data(),
           sizeof(int64_t) * detection_result->masks[i].shape.size());
    fd_c_detection_result->masks.data[i].type =
        static_cast<FD_C_ResultType>(detection_result->masks[i].type);
  }
  fd_c_detection_result->contain_masks = detection_result->contain_masks;
  fd_c_detection_result->type =
      static_cast<FD_C_ResultType>(detection_result->type);
}

FD_C_DetectionResultWrapper* FD_C_CreateDetectionResultWrapperFromCResult(
    __fd_keep FD_C_DetectionResult* fd_c_detection_result) {
  FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper =
      FD_C_CreateDetectionResultWrapper();
  auto& detection_result = CHECK_AND_CONVERT_FD_TYPE(
      DetectionResultWrapper, fd_c_detection_result_wrapper);

  // copy boxes
  const int boxes_coordinate_dim = 4;
  detection_result->boxes.resize(fd_c_detection_result->boxes.size);
  for (size_t i = 0; i < fd_c_detection_result->boxes.size; i++) {
    for (size_t j = 0; j < boxes_coordinate_dim; j++) {
      detection_result->boxes[i][j] =
          fd_c_detection_result->boxes.data[i].data[j];
    }
  }
  // copy rotated_boxes
  const int rotated_boxes_coordinate_dim = 8;
  detection_result->rotated_boxes.resize(
      fd_c_detection_result->rotated_boxes.size);
  for (size_t i = 0; i < fd_c_detection_result->rotated_boxes.size; i++) {
    for (size_t j = 0; j < rotated_boxes_coordinate_dim; j++) {
      detection_result->rotated_boxes[i][j] =
          fd_c_detection_result->rotated_boxes.data[i].data[j];
    }
  }
  // copy scores
  detection_result->scores.resize(fd_c_detection_result->scores.size);
  memcpy(detection_result->scores.data(), fd_c_detection_result->scores.data,
         sizeof(float) * fd_c_detection_result->scores.size);
  // copy label_ids
  detection_result->label_ids.resize(fd_c_detection_result->label_ids.size);
  memcpy(detection_result->label_ids.data(),
         fd_c_detection_result->label_ids.data,
         sizeof(int32_t) * fd_c_detection_result->label_ids.size);
  // copy masks
  detection_result->masks.resize(fd_c_detection_result->masks.size);
  for (size_t i = 0; i < fd_c_detection_result->masks.size; i++) {
    // copy data in mask
    detection_result->masks[i].data.resize(
        fd_c_detection_result->masks.data[i].data.size);
    memcpy(detection_result->masks[i].data.data(),
           fd_c_detection_result->masks.data[i].data.data,
           sizeof(uint8_t) * fd_c_detection_result->masks.data[i].data.size);
    // copy shape in mask
    detection_result->masks[i].shape.resize(
        fd_c_detection_result->masks.data[i].shape.size);
    memcpy(detection_result->masks[i].shape.data(),
           fd_c_detection_result->masks.data[i].shape.data,
           sizeof(int64_t) * fd_c_detection_result->masks.data[i].shape.size);
    detection_result->masks[i].type =
        static_cast<fastdeploy::vision::ResultType>(
            fd_c_detection_result->masks.data[i].type);
  }
  detection_result->contain_masks = fd_c_detection_result->contain_masks;
  detection_result->type =
      static_cast<fastdeploy::vision::ResultType>(fd_c_detection_result->type);

  return fd_c_detection_result_wrapper;
}

void FD_C_DetectionResultStr(FD_C_DetectionResult* fd_c_detection_result,
                             char* str_buffer) {
  FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper =
      FD_C_CreateDetectionResultWrapperFromCResult(fd_c_detection_result);
  auto& detection_result = CHECK_AND_CONVERT_FD_TYPE(
      DetectionResultWrapper, fd_c_detection_result_wrapper);
  std::string information = detection_result->Str();
  std::strcpy(str_buffer, information.c_str());
  FD_C_DestroyDetectionResultWrapper(fd_c_detection_result_wrapper);
}

// OCR Results

FD_C_OCRResultWrapper* FD_C_CreateOCRResultWrapper() {
  FD_C_OCRResultWrapper* fd_c_ocr_result_wrapper = new FD_C_OCRResultWrapper();
  fd_c_ocr_result_wrapper->ocr_result =
      std::unique_ptr<fastdeploy::vision::OCRResult>(
          new fastdeploy::vision::OCRResult());
  return fd_c_ocr_result_wrapper;
}

void FD_C_DestroyOCRResultWrapper(
    __fd_take FD_C_OCRResultWrapper* fd_c_ocr_result_wrapper) {
  delete fd_c_ocr_result_wrapper;
}

FD_C_OCRResult* FD_C_CreateOCRResult() {
  FD_C_OCRResult* fd_c_ocr_result = new FD_C_OCRResult();
  return fd_c_ocr_result;
}

void FD_C_DestroyOCRResultInner(__fd_take FD_C_OCRResult* fd_c_ocr_result) {
  if (fd_c_ocr_result == nullptr)
    return;
  // delete boxes
  for (size_t i = 0; i < fd_c_ocr_result->boxes.size; i++) {
    delete[] fd_c_ocr_result->boxes.data[i].data;
  }
  delete[] fd_c_ocr_result->boxes.data;
  // delete text
  for (size_t i = 0; i < fd_c_ocr_result->text.size; i++) {
    delete[] fd_c_ocr_result->text.data[i].data;
  }
  delete[] fd_c_ocr_result->text.data;
  // delete rec_scores
  delete[] fd_c_ocr_result->rec_scores.data;
  // delete cls_scores
  delete[] fd_c_ocr_result->cls_scores.data;
  // delete cls_labels
  delete[] fd_c_ocr_result->cls_labels.data;
  // delete table_boxes
  for (size_t i = 0; i < fd_c_ocr_result->table_boxes.size; i++) {
    delete[] fd_c_ocr_result->table_boxes.data[i].data;
  }
  delete[] fd_c_ocr_result->table_boxes.data;
  // delete table_structure
  for (size_t i = 0; i < fd_c_ocr_result->table_structure.size; i++) {
    delete[] fd_c_ocr_result->table_structure.data[i].data;
  }
  delete[] fd_c_ocr_result->table_structure.data;
  // delete table_html
  delete[] fd_c_ocr_result->table_html.data;
}

void FD_C_DestroyOCRResult(__fd_take FD_C_OCRResult* fd_c_ocr_result) {
  if (fd_c_ocr_result == nullptr)
    return;
  FD_C_DestroyOCRResultInner(fd_c_ocr_result);
  delete fd_c_ocr_result;
}

FD_C_OneDimOCRResult* FD_C_CreateOneDimOCRResult() {
  FD_C_OneDimOCRResult* fd_c_one_dim_ocr_result = new FD_C_OneDimOCRResult();
  return fd_c_one_dim_ocr_result;
}

void FD_C_DestroyOneDimOCRResultInner(
    __fd_take FD_C_OneDimOCRResult* fd_c_one_dim_ocr_result) {
  if (fd_c_one_dim_ocr_result == nullptr) {
    return;
  }
  for (int i = 0; i < fd_c_one_dim_ocr_result->size; i++) {
    FD_C_DestroyOCRResultInner(fd_c_one_dim_ocr_result->data + i);
  }
  delete[] fd_c_one_dim_ocr_result->data;
}

void FD_C_DestroyOneDimOCRResult(
    __fd_take FD_C_OneDimOCRResult* fd_c_one_dim_ocr_result) {
  FD_C_DestroyOneDimOCRResultInner(fd_c_one_dim_ocr_result);
  delete fd_c_one_dim_ocr_result;
}

void FD_C_OCRResultWrapperToCResult(
    __fd_keep FD_C_OCRResultWrapper* fd_c_ocr_result_wrapper,
    __fd_keep FD_C_OCRResult* fd_c_ocr_result) {
  auto& ocr_result =
      CHECK_AND_CONVERT_FD_TYPE(OCRResultWrapper, fd_c_ocr_result_wrapper);
  // copy boxes
  const int boxes_coordinate_dim = 8;
  fd_c_ocr_result->boxes.size = ocr_result->boxes.size();
  fd_c_ocr_result->boxes.data =
      new FD_C_OneDimArrayInt32[fd_c_ocr_result->boxes.size];
  for (size_t i = 0; i < ocr_result->boxes.size(); i++) {
    fd_c_ocr_result->boxes.data[i].size = boxes_coordinate_dim;
    fd_c_ocr_result->boxes.data[i].data = new int[boxes_coordinate_dim];
    for (size_t j = 0; j < boxes_coordinate_dim; j++) {
      fd_c_ocr_result->boxes.data[i].data[j] = ocr_result->boxes[i][j];
    }
  }
  // copy text
  fd_c_ocr_result->text.size = ocr_result->text.size();
  fd_c_ocr_result->text.data = new FD_C_Cstr[fd_c_ocr_result->text.size];
  for (size_t i = 0; i < ocr_result->text.size(); i++) {
    fd_c_ocr_result->text.data[i].size = ocr_result->text[i].length();
    fd_c_ocr_result->text.data[i].data =
        new char[ocr_result->text[i].length() + 1];
    strcpy(fd_c_ocr_result->text.data[i].data, ocr_result->text[i].c_str());
  }

  // copy rec_scores
  fd_c_ocr_result->rec_scores.size = ocr_result->rec_scores.size();
  fd_c_ocr_result->rec_scores.data =
      new float[fd_c_ocr_result->rec_scores.size];
  memcpy(fd_c_ocr_result->rec_scores.data, ocr_result->rec_scores.data(),
         sizeof(float) * fd_c_ocr_result->rec_scores.size);
  // copy cls_scores
  fd_c_ocr_result->cls_scores.size = ocr_result->cls_scores.size();
  fd_c_ocr_result->cls_scores.data =
      new float[fd_c_ocr_result->cls_scores.size];
  memcpy(fd_c_ocr_result->cls_scores.data, ocr_result->cls_scores.data(),
         sizeof(float) * fd_c_ocr_result->cls_scores.size);
  // copy cls_labels
  fd_c_ocr_result->cls_labels.size = ocr_result->cls_labels.size();
  fd_c_ocr_result->cls_labels.data =
      new int32_t[fd_c_ocr_result->cls_labels.size];
  memcpy(fd_c_ocr_result->cls_labels.data, ocr_result->cls_labels.data(),
         sizeof(int32_t) * fd_c_ocr_result->cls_labels.size);

  // copy table boxes
  fd_c_ocr_result->table_boxes.size = ocr_result->table_boxes.size();
  fd_c_ocr_result->table_boxes.data =
      new FD_C_OneDimArrayInt32[fd_c_ocr_result->table_boxes.size];
  for (size_t i = 0; i < ocr_result->table_boxes.size(); i++) {
    fd_c_ocr_result->table_boxes.data[i].size = boxes_coordinate_dim;
    fd_c_ocr_result->table_boxes.data[i].data = new int[boxes_coordinate_dim];
    for (size_t j = 0; j < boxes_coordinate_dim; j++) {
      fd_c_ocr_result->table_boxes.data[i].data[j] =
          ocr_result->table_boxes[i][j];
    }
  }

  // copy table_structure
  fd_c_ocr_result->table_structure.size = ocr_result->table_structure.size();
  fd_c_ocr_result->table_structure.data =
      new FD_C_Cstr[fd_c_ocr_result->table_structure.size];
  for (size_t i = 0; i < ocr_result->table_structure.size(); i++) {
    fd_c_ocr_result->table_structure.data[i].size =
        ocr_result->text[i].length();
    fd_c_ocr_result->table_structure.data[i].data =
        new char[ocr_result->table_structure[i].length() + 1];
    strcpy(fd_c_ocr_result->table_structure.data[i].data,
           ocr_result->table_structure[i].c_str());
  }

  // copy table_html
  fd_c_ocr_result->table_html.data =
      new char[ocr_result->table_html.length() + 1];
  strcpy(fd_c_ocr_result->table_html.data, ocr_result->table_html.c_str());

  // copy type
  fd_c_ocr_result->type = static_cast<FD_C_ResultType>(ocr_result->type);
}

FD_C_OCRResultWrapper* FD_C_CreateOCRResultWrapperFromCResult(
    __fd_keep FD_C_OCRResult* fd_c_ocr_result) {
  FD_C_OCRResultWrapper* fd_c_ocr_result_wrapper =
      FD_C_CreateOCRResultWrapper();
  auto& ocr_result =
      CHECK_AND_CONVERT_FD_TYPE(OCRResultWrapper, fd_c_ocr_result_wrapper);

  // copy boxes
  const int boxes_coordinate_dim = 8;
  ocr_result->boxes.resize(fd_c_ocr_result->boxes.size);
  for (size_t i = 0; i < fd_c_ocr_result->boxes.size; i++) {
    for (size_t j = 0; j < boxes_coordinate_dim; j++) {
      ocr_result->boxes[i][j] = fd_c_ocr_result->boxes.data[i].data[j];
    }
  }

  // copy text
  ocr_result->text.resize(fd_c_ocr_result->text.size);
  for (size_t i = 0; i < ocr_result->text.size(); i++) {
    ocr_result->text[i] = std::string(fd_c_ocr_result->text.data[i].data);
  }

  // copy rec_scores
  ocr_result->rec_scores.resize(fd_c_ocr_result->rec_scores.size);
  memcpy(ocr_result->rec_scores.data(), fd_c_ocr_result->rec_scores.data,
         sizeof(float) * fd_c_ocr_result->rec_scores.size);

  // copy cls_scores
  ocr_result->cls_scores.resize(fd_c_ocr_result->cls_scores.size);
  memcpy(ocr_result->cls_scores.data(), fd_c_ocr_result->cls_scores.data,
         sizeof(float) * fd_c_ocr_result->cls_scores.size);

  // copy cls_labels
  ocr_result->cls_labels.resize(fd_c_ocr_result->cls_labels.size);
  memcpy(ocr_result->cls_labels.data(), fd_c_ocr_result->cls_labels.data,
         sizeof(int32_t) * fd_c_ocr_result->cls_labels.size);

  // copy type
  ocr_result->type =
      static_cast<fastdeploy::vision::ResultType>(fd_c_ocr_result->type);

  return fd_c_ocr_result_wrapper;
}

void FD_C_OCRResultStr(FD_C_OCRResult* fd_c_ocr_result, char* str_buffer) {
  FD_C_OCRResultWrapper* fd_c_ocr_result_wrapper =
      FD_C_CreateOCRResultWrapperFromCResult(fd_c_ocr_result);
  auto& ocr_result =
      CHECK_AND_CONVERT_FD_TYPE(OCRResultWrapper, fd_c_ocr_result_wrapper);
  std::string information = ocr_result->Str();
  std::strcpy(str_buffer, information.c_str());
  FD_C_DestroyOCRResultWrapper(fd_c_ocr_result_wrapper);
  return;
}

// Segmentation Results

FD_C_SegmentationResultWrapper* FD_C_CreateSegmentationResultWrapper() {
  FD_C_SegmentationResultWrapper* fd_c_segmentation_result_wrapper =
      new FD_C_SegmentationResultWrapper();
  fd_c_segmentation_result_wrapper->segmentation_result =
      std::unique_ptr<fastdeploy::vision::SegmentationResult>(
          new fastdeploy::vision::SegmentationResult());
  return fd_c_segmentation_result_wrapper;
}

void FD_C_DestroySegmentationResultWrapper(
    __fd_take FD_C_SegmentationResultWrapper*
        fd_c_segmentation_result_wrapper) {
  delete fd_c_segmentation_result_wrapper;
}

FD_C_SegmentationResult* FD_C_CreateSegmentationResult() {
  FD_C_SegmentationResult* fd_c_segmentation_result =
      new FD_C_SegmentationResult();
  return fd_c_segmentation_result;
}

FD_C_OneDimSegmentationResult* FD_C_CreateOneDimSegmentationResult() {
  FD_C_OneDimSegmentationResult* fd_c_one_dim_segmentation_result =
      new FD_C_OneDimSegmentationResult();
  return fd_c_one_dim_segmentation_result;
}

void FD_C_DestroySegmentationResultInner(
    __fd_take FD_C_SegmentationResult* fd_c_segmentation_result) {
  if (fd_c_segmentation_result == nullptr)
    return;
  // delete label_map
  delete[] fd_c_segmentation_result->label_map.data;
  // delete score_map
  delete[] fd_c_segmentation_result->score_map.data;
  // delete shape
  delete[] fd_c_segmentation_result->shape.data;
}

void FD_C_DestroySegmentationResult(
    __fd_take FD_C_SegmentationResult* fd_c_segmentation_result) {
  if (fd_c_segmentation_result == nullptr)
    return;
  FD_C_DestroySegmentationResultInner(fd_c_segmentation_result);
  delete fd_c_segmentation_result;
}

void FD_C_DestroyOneDimSegmentationResultInner(
    FD_C_OneDimSegmentationResult* fd_c_one_dim_segmentation_result) {
  if (fd_c_one_dim_segmentation_result == nullptr)
    return;
  for (int i = 0; i < fd_c_one_dim_segmentation_result->size; i++) {
    FD_C_DestroySegmentationResultInner(fd_c_one_dim_segmentation_result->data +
                                        i);
  }
  delete[] fd_c_one_dim_segmentation_result->data;
}

void FD_C_DestroyOneDimSegmentationResult(
    FD_C_OneDimSegmentationResult* fd_c_one_dim_segmentation_result) {
  if (fd_c_one_dim_segmentation_result == nullptr)
    return;
  FD_C_DestroyOneDimSegmentationResultInner(fd_c_one_dim_segmentation_result);
  delete fd_c_one_dim_segmentation_result;
}

void FD_C_SegmentationResultWrapperToCResult(
    __fd_keep FD_C_SegmentationResultWrapper* fd_c_segmentation_result_wrapper,
    __fd_keep FD_C_SegmentationResult* fd_c_segmentation_result) {
  auto& segmentation_result = CHECK_AND_CONVERT_FD_TYPE(
      SegmentationResultWrapper, fd_c_segmentation_result_wrapper);

  // copy label_map
  fd_c_segmentation_result->label_map.size =
      segmentation_result->label_map.size();
  fd_c_segmentation_result->label_map.data =
      new uint8_t[fd_c_segmentation_result->label_map.size];
  memcpy(fd_c_segmentation_result->label_map.data,
         segmentation_result->label_map.data(),
         sizeof(uint8_t) * fd_c_segmentation_result->label_map.size);
  // copy score_map
  fd_c_segmentation_result->score_map.size =
      segmentation_result->score_map.size();
  fd_c_segmentation_result->score_map.data =
      new float[fd_c_segmentation_result->score_map.size];
  memcpy(fd_c_segmentation_result->score_map.data,
         segmentation_result->score_map.data(),
         sizeof(float) * fd_c_segmentation_result->score_map.size);
  // copy shape
  fd_c_segmentation_result->shape.size = segmentation_result->shape.size();
  fd_c_segmentation_result->shape.data =
      new int64_t[fd_c_segmentation_result->shape.size];
  memcpy(fd_c_segmentation_result->shape.data,
         segmentation_result->shape.data(),
         sizeof(int64_t) * fd_c_segmentation_result->shape.size);
  // copy contain_score_map
  fd_c_segmentation_result->contain_score_map =
      segmentation_result->contain_score_map;
  // copy type
  fd_c_segmentation_result->type =
      static_cast<FD_C_ResultType>(segmentation_result->type);
}

FD_C_SegmentationResultWrapper* FD_C_CreateSegmentationResultWrapperFromCResult(
    __fd_keep FD_C_SegmentationResult* fd_c_segmentation_result) {
  FD_C_SegmentationResultWrapper* fd_c_segmentation_result_wrapper =
      FD_C_CreateSegmentationResultWrapper();
  auto& segmentation_result = CHECK_AND_CONVERT_FD_TYPE(
      SegmentationResultWrapper, fd_c_segmentation_result_wrapper);

  // copy label_map
  segmentation_result->label_map.resize(
      fd_c_segmentation_result->label_map.size);
  memcpy(segmentation_result->label_map.data(),
         fd_c_segmentation_result->label_map.data,
         sizeof(uint8_t) * fd_c_segmentation_result->label_map.size);

  // copy score_map
  segmentation_result->score_map.resize(
      fd_c_segmentation_result->score_map.size);
  memcpy(segmentation_result->score_map.data(),
         fd_c_segmentation_result->score_map.data,
         sizeof(float) * fd_c_segmentation_result->score_map.size);

  // copy shape
  segmentation_result->shape.resize(fd_c_segmentation_result->shape.size);
  memcpy(segmentation_result->shape.data(),
         fd_c_segmentation_result->shape.data,
         sizeof(int64_t) * fd_c_segmentation_result->shape.size);

  // copy contain_score_map
  segmentation_result->contain_score_map =
      fd_c_segmentation_result->contain_score_map;
  // copy type
  segmentation_result->type = static_cast<fastdeploy::vision::ResultType>(
      fd_c_segmentation_result->type);

  return fd_c_segmentation_result_wrapper;
}

void FD_C_SegmentationResultStr(
    FD_C_SegmentationResult* fd_c_segmentation_result, char* str_buffer) {
  FD_C_SegmentationResultWrapper* fd_c_segmentation_result_wrapper =
      FD_C_CreateSegmentationResultWrapperFromCResult(fd_c_segmentation_result);
  auto& segmentation_result = CHECK_AND_CONVERT_FD_TYPE(
      SegmentationResultWrapper, fd_c_segmentation_result_wrapper);
  std::string information = segmentation_result->Str();
  std::strcpy(str_buffer, information.c_str());
  FD_C_DestroySegmentationResultWrapper(fd_c_segmentation_result_wrapper);
  return;
}

#ifdef __cplusplus
}
#endif
