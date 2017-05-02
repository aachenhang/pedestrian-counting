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
	//   �ҵ���һ�������һ����0��ɫ��ֵ
	for (First = 0; First < HistGram.size() && HistGram[First] == 0; First++);
	for (Last = HistGram.size() - 1; Last > First && HistGram[Last] == 0; Last--);
	if (First == Last) return First;                // ͼ����ֻ��һ����ɫ
	if (First + 1 == Last) return First;            // ͼ����ֻ�ж�����ɫ

													// �����ۼ�ֱ��ͼ�Լ���Ӧ�Ĵ�Ȩ�ص��ۼ�ֱ��ͼ
	int* S = new int[Last + 1];
	int* W = new int[Last + 1];            // �����ش�ͼ��������ı������ݿ��ܻᳬ��int�ı�ʾ��Χ�����Կ�����long����������
	S[0] = HistGram[0];
	for (Y = First > 1 ? First : 1; Y <= Last; Y++)
	{
		S[Y] = S[Y - 1] + HistGram[Y];
		W[Y] = W[Y - 1] + Y * HistGram[Y];
	}

	// ������ʽ��4������6�����õĲ��ұ�
	vector<double> Smu(Last + 1 - First);
	for (Y = 1; Y < Smu.size(); Y++)
	{
		double mu = 1 / (1 + (double)Y / (Last - First));               // ��ʽ��4��
		Smu[Y] = -mu * log(mu) - (1 - mu) * log(1 - mu);      // ��ʽ��6��
	}

	// �������������ֵ
	for (Y = First; Y <= Last; Y++)
	{
		Entropy = 0;
		int mu = (int)round((double)W[Y] / S[Y]);             // ��ʽ17
		for (X = First; X <= Y; X++)
			Entropy += Smu[abs(X - mu)] * HistGram[X];
		mu = (int)round((double)(W[Last] - W[Y]) / (S[Last] - S[Y]));  // ��ʽ18
		for (X = Y + 1; X <= Last; X++)
			Entropy += Smu[abs(X - mu)] * HistGram[X];       // ��ʽ8
		if (BestEntropy > Entropy)
		{
			BestEntropy = Entropy;      // ȡ��С�ش�Ϊ�����ֵ
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

	for (Y = 0; Y <= 255; Y++) Amount += HistGram[Y];        //  ��������
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

	if (MaxValue == MinValue) return MaxValue;          // ͼ����ֻ��һ����ɫ             
	if (MinValue + 1 == MaxValue) return MinValue;      // ͼ����ֻ�ж�����ɫ

	Threshold = MinValue;
	NewThreshold = (MaxValue + MinValue) >> 1;
	while (Threshold != NewThreshold)    // ��ǰ�����ε����Ļ����ֵ��ͬʱ����������    
	{
		SumOne = 0; SumIntegralOne = 0;
		SumTwo = 0; SumIntegralTwo = 0;
		Threshold = NewThreshold;
		for (X = MinValue; X <= Threshold; X++)         //������ֵ��ͼ��ָ��Ŀ��ͱ��������֣���������ֵ�ƽ���Ҷ�ֵ      
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
		NewThreshold = (MeanValueOne + MeanValueTwo) >> 1;       //����µ���ֵ
		Iter++;
		if (Iter >= 1000) return -1;
	}
	return Threshold;
}

int GetOSTUThreshold(const vector<int> &HistGram)
{
	int X, Y, Amount = 0;
	int PixelBack = 0, PixelFore = 0, PixelIntegralBack = 0, PixelIntegralFore = 0, PixelIntegral = 0;
	double OmegaBack, OmegaFore, MicroBack, MicroFore, SigmaB, Sigma;              // ��䷽��;
	int MinValue, MaxValue;
	int Threshold = 0;

	for (MinValue = 0; MinValue < 256 && HistGram[MinValue] == 0; MinValue++);
	for (MaxValue = 255; MaxValue > MinValue && HistGram[MinValue] == 0; MaxValue--);
	if (MaxValue == MinValue) return MaxValue;          // ͼ����ֻ��һ����ɫ             
	if (MinValue + 1 == MaxValue) return MinValue;      // ͼ����ֻ�ж�����ɫ

	for (Y = MinValue; Y <= MaxValue; Y++) Amount += HistGram[Y];        //  ��������

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
