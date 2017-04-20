#pragma once

/*
Copyright (c) 2013, Taiga Nomi
All rights reserved.

Use of this source code is governed by a BSD-style license that can be found
in the LICENSE file.
*/

#include <opencv2/imgproc.hpp>



void sample1_convnet(float alpha = 0.01);

void convnet_test(String imgFileName = image_test_file, double maxv = 1.0, double minv = -1.0);
