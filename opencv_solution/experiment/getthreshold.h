#pragma once
#include <vector>
#include <limits>
#include <cmath>
using namespace std;

static double A(const vector<int> &HistGram, int Index)
{
	double Sum = 0;
	for (int Y = 0; Y <= Index; Y++)
		Sum += HistGram[Y];
	return Sum;
}

static double B(const vector<int> &HistGram, int Index)
{
	double Sum = 0;
	for (int Y = 0; Y <= Index; Y++)
		Sum += (double)Y * HistGram[Y];
	return Sum;
}

static double C(const vector<int> &HistGram, int Index)
{
	double Sum = 0;
	for (int Y = 0; Y <= Index; Y++)
		Sum += (double)Y * Y * HistGram[Y];
	return Sum;
}

static double D(const vector<int> &HistGram, int Index)
{
	double Sum = 0;
	for (int Y = 0; Y <= Index; Y++)
		Sum += (double)Y * Y * Y * HistGram[Y];
	return Sum;
}

namespace Thre {

int GetHuangFuzzyThreshold(const vector<int> &HistGram)
{
	int X, Y;
	int First, Last;
	int Threshold = -1;
	double BestEntropy = INT_MAX, Entropy;
	//   找到第一个和最后一个非0的色阶值
	for (First = 0; First < HistGram.size() && HistGram[First] == 0; First++);
	for (Last = HistGram.size() - 1; Last > First && HistGram[Last] == 0; Last--);
	if (First == Last) return First;                // 图像中只有一个颜色
	if (First + 1 == Last) return First;            // 图像中只有二个颜色

													// 计算累计直方图以及对应的带权重的累计直方图
	int* S = new int[Last + 1];
	int* W = new int[Last + 1];            // 对于特大图，此数组的保存数据可能会超出int的表示范围，可以考虑用long类型来代替
	S[0] = HistGram[0];
	for (Y = First > 1 ? First : 1; Y <= Last; Y++)
	{
		S[Y] = S[Y - 1] + HistGram[Y];
		W[Y] = W[Y - 1] + Y * HistGram[Y];
	}

	// 建立公式（4）及（6）所用的查找表
	vector<double> Smu(Last + 1 - First);
	for (Y = 1; Y < Smu.size(); Y++)
	{
		double mu = 1 / (1 + (double)Y / (Last - First));               // 公式（4）
		Smu[Y] = -mu * log(mu) - (1 - mu) * log(1 - mu);      // 公式（6）
	}

	// 迭代计算最佳阈值
	for (Y = First; Y <= Last; Y++)
	{
		Entropy = 0;
		int mu = (int)round((double)W[Y] / S[Y]);             // 公式17
		for (X = First; X <= Y; X++)
			Entropy += Smu[abs(X - mu)] * HistGram[X];
		mu = (int)round((double)(W[Last] - W[Y]) / (S[Last] - S[Y]));  // 公式18
		for (X = Y + 1; X <= Last; X++)
			Entropy += Smu[abs(X - mu)] * HistGram[X];       // 公式8
		if (BestEntropy > Entropy)
		{
			BestEntropy = Entropy;      // 取最小熵处为最佳阈值
			Threshold = Y;
		}
	}
	return Threshold;
}

int GetMomentPreservingThreshold(const vector<int> &HistGram)
{
	int X, Y, Index = 0, Amount = 0;
	double* Avec = new double[256];
	double X2, X1, X0, Min;

	for (Y = 0; Y <= 255; Y++) Amount += HistGram[Y];        //  像素总数
	for (Y = 0; Y < 256; Y++) Avec[Y] = (double)A(HistGram, Y) / Amount;       // The threshold is chosen such that A(y,t)/A(y,n) is closest to x0.

																				// The following finds x0.

	X2 = (double)(B(HistGram, 255) * C(HistGram, 255) - A(HistGram, 255) * D(HistGram, 255)) / (double)(A(HistGram, 255) * C(HistGram, 255) - B(HistGram, 255) * B(HistGram, 255));
	X1 = (double)(B(HistGram, 255) * D(HistGram, 255) - C(HistGram, 255) * C(HistGram, 255)) / (double)(A(HistGram, 255) * C(HistGram, 255) - B(HistGram, 255) * B(HistGram, 255));
	X0 = 0.5 - (B(HistGram, 255) / A(HistGram, 255) + X2 / 2) / sqrt(X2 * X2 - 4 * X1);

	for (Y = 0, Min = INT_MAX; Y < 256; Y++)
	{
		if (abs(Avec[Y] - X0) < Min)
		{
			Min = abs(Avec[Y] - X0);
			Index = Y;
		}
	}
	return Index;
}

int GetMeanThreshold(const vector<int> &HistGram)
{
	int Sum = 0, Amount = 0;
	for (int Y = 0; Y < 256; Y++)
	{
		Amount += HistGram[Y];
		Sum += Y * HistGram[Y];
	}
	return Sum / Amount;
}

int GetIterativeBestThreshold(const vector<int> &HistGram)
{
	int X, Iter = 0;
	int MeanValueOne, MeanValueTwo, SumOne, SumTwo, SumIntegralOne, SumIntegralTwo;
	int MinValue, MaxValue;
	int Threshold, NewThreshold;

	for (MinValue = 0; MinValue < 256 && HistGram[MinValue] == 0; MinValue++);
	for (MaxValue = 255; MaxValue > MinValue && HistGram[MinValue] == 0; MaxValue--);

	if (MaxValue == MinValue) return MaxValue;          // 图像中只有一个颜色             
	if (MinValue + 1 == MaxValue) return MinValue;      // 图像中只有二个颜色

	Threshold = MinValue;
	NewThreshold = (MaxValue + MinValue) >> 1;
	while (Threshold != NewThreshold)    // 当前后两次迭代的获得阈值相同时，结束迭代    
	{
		SumOne = 0; SumIntegralOne = 0;
		SumTwo = 0; SumIntegralTwo = 0;
		Threshold = NewThreshold;
		for (X = MinValue; X <= Threshold; X++)         //根据阈值将图像分割成目标和背景两部分，求出两部分的平均灰度值      
		{
			SumIntegralOne += HistGram[X] * X;
			SumOne += HistGram[X];
		}
		MeanValueOne = SumIntegralOne / SumOne;
		for (X = Threshold + 1; X <= MaxValue; X++)
		{
			SumIntegralTwo += HistGram[X] * X;
			SumTwo += HistGram[X];
		}
		MeanValueTwo = SumIntegralTwo / SumTwo;
		NewThreshold = (MeanValueOne + MeanValueTwo) >> 1;       //求出新的阈值
		Iter++;
		if (Iter >= 1000) return -1;
	}
	return Threshold;
}

int GetOSTUThreshold(const vector<int> &HistGram)
{
	int X, Y, Amount = 0;
	int PixelBack = 0, PixelFore = 0, PixelIntegralBack = 0, PixelIntegralFore = 0, PixelIntegral = 0;
	double OmegaBack, OmegaFore, MicroBack, MicroFore, SigmaB, Sigma;              // 类间方差;
	int MinValue, MaxValue;
	int Threshold = 0;

	for (MinValue = 0; MinValue < 256 && HistGram[MinValue] == 0; MinValue++);
	for (MaxValue = 255; MaxValue > MinValue && HistGram[MinValue] == 0; MaxValue--);
	if (MaxValue == MinValue) return MaxValue;          // 图像中只有一个颜色             
	if (MinValue + 1 == MaxValue) return MinValue;      // 图像中只有二个颜色

	for (Y = MinValue; Y <= MaxValue; Y++) Amount += HistGram[Y];        //  像素总数

	PixelIntegral = 0;
	for (Y = MinValue; Y <= MaxValue; Y++) PixelIntegral += HistGram[Y] * Y;
	SigmaB = -1;
	for (Y = MinValue; Y < MaxValue; Y++)
	{
		PixelBack = PixelBack + HistGram[Y];
		PixelFore = Amount - PixelBack;
		OmegaBack = (double)PixelBack / Amount;
		OmegaFore = (double)PixelFore / Amount;
		PixelIntegralBack += HistGram[Y] * Y;
		PixelIntegralFore = PixelIntegral - PixelIntegralBack;
		MicroBack = (double)PixelIntegralBack / PixelBack;
		MicroFore = (double)PixelIntegralFore / PixelFore;
		Sigma = OmegaBack * OmegaFore * (MicroBack - MicroFore) * (MicroBack - MicroFore);
		if (Sigma > SigmaB)
		{
			SigmaB = Sigma;
			Threshold = Y;
		}
	}
	return Threshold;
}

// M. Emre Celebi
// 06.15.2007
// Ported to ImageJ plugin by G.Landini from E Celebi's fourier_0.8 routines
int GetYenThreshold(const vector<int> &HistGram)
{
	int threshold;
	int ih, it;
	double crit;
	double max_crit;
	double* norm_histo = new double[HistGram.size()]; /* normalized histogram */
	double* P1 = new double[HistGram.size()]; /* cumulative normalized histogram */
	double* P1_sq = new double[HistGram.size()];
	double* P2_sq = new double[HistGram.size()];

	int total = 0;
	for (ih = 0; ih < HistGram.size(); ih++)
		total += HistGram[ih];

	for (ih = 0; ih < HistGram.size(); ih++)
		norm_histo[ih] = (double)HistGram[ih] / total;

	P1[0] = norm_histo[0];
	for (ih = 1; ih < HistGram.size(); ih++)
		P1[ih] = P1[ih - 1] + norm_histo[ih];

	P1_sq[0] = norm_histo[0] * norm_histo[0];
	for (ih = 1; ih < HistGram.size(); ih++)
		P1_sq[ih] = P1_sq[ih - 1] + norm_histo[ih] * norm_histo[ih];

	P2_sq[HistGram.size() - 1] = 0.0;
	for (ih = HistGram.size() - 2; ih >= 0; ih--)
		P2_sq[ih] = P2_sq[ih + 1] + norm_histo[ih + 1] * norm_histo[ih + 1];

	/* Find the threshold that maximizes the criterion */
	threshold = -1;
	max_crit = INT_MIN;
	for (it = 0; it < HistGram.size(); it++)
	{
		crit = -1.0 * ((P1_sq[it] * P2_sq[it]) > 0.0 ? log(P1_sq[it] * P2_sq[it]) : 0.0) + 2 * ((P1[it] * (1.0 - P1[it])) > 0.0 ? log(P1[it] * (1.0 - P1[it])) : 0.0);
		if (crit > max_crit)
		{
			max_crit = crit;
			threshold = it;
		}
	}
	return threshold;
}
}
