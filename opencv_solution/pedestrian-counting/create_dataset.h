#pragma once


void mouseClick(int event, int x, int y, int flags, void* userdata);

int makeAnnotation(int inputnum);

int catchimage(int inputnum);

int cropimage(int inputnum);

int createNegativeSample(int inputnum, int sum);

void createHardSample();