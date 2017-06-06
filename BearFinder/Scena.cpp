#include "Scena.h"
#ifdef Debug
Mat tempzd;
int Scena::nrzd(0);
#endif

Scena::Scena()
{
#ifdef Debug
	
#endif
}


Scena::~Scena()
{
}

void Scena::dodajUjecie(std::string s)
{
	Mat zdj;
	zdj=imread(s, CV_LOAD_IMAGE_COLOR);
	if (zdj.empty())
		return;
#ifdef Debug
	tempzd = zdj.clone();
#endif
	rozmiarObrazu = zdj.size();
	vector<vector<Point>> ksztalty = wykryjKsztalty(zdj);
	vector<Point> p = znajdzWierzcholki(ksztalty);
	if (p.empty())
		return;
	Rect BRect;
	BRect=boundingRect(Mat(p));
	Mat mask;
	mask.create(zdj.size(), CV_8UC1);
	mask.setTo(0);
	fillConvexPoly(mask, p, Scalar(255));
	Mat maskowane;
	zdj.copyTo(maskowane, mask);
	//zdj = maskowane(BRect);
	//SimplestCB(zdj, zdj, 1);
	medianBlur(zdj, zdj, 13);
	rzutuj(zdj, p);
	//for (auto &punkt : p)
	//	punkt = punkt - Point(BRect.x, BRect.y);

	//Mat matNormalizacji;
	//matNormalizacji.create(zdj.size(), CV_32FC1);
	//normalizuj(p, matNormalizacji, znacznik);
	policzZelki(zdj, p);
	cout << " ok" << endl;
	return;
}

void kuwahara1c(Mat in_out, int k)
{
	Mat temp = in_out.clone();
	Mat p0(k, k, in_out.type());
	Mat p1(k, k, in_out.type());
	Mat p2(k, k, in_out.type());
	Mat p3(k, k, in_out.type());
	int row = temp.rows;
	int col = temp.cols;
	for (int i = 0; i < row; ++i){
		for (int j = 0; j < col; j++){
			int x1 = j - k;
			int y1 = i - k;
			Rect r0(x1, y1, k, k);
			Rect r1(j, y1, k, k);
			Rect r2(j, i, k, k);
			Rect r3(x1, i, k, k);
			Mat m,d;
			int mean = 1000,dev=1000;
			if (x1>0 && y1 >= 0){
				meanStdDev(temp(r0), m, d);
				if (d.at<double>(0, 0) < dev)
				{
					dev = d.at<double>(0,0);
					mean = m.at<double>(0, 0);
				}
			}
			if (j + k < col && y1 >= 0){
				meanStdDev(temp(r1), m, d);
				if (d.at<double>(0, 0) < dev)
				{
					dev = d.at<double>(0, 0);
					mean = m.at<double>(0, 0);
				}
			}
			if (i + k < row && j + k < col){
				meanStdDev(temp(r2), m, d);
				if (d.at<double>(0, 0) < dev)
				{
					dev = d.at<double>(0, 0);
					mean = m.at<double>(0, 0);
				}
			}
			if (x1 >= 0 && i + k < row){
				meanStdDev(temp(r3), m, d);
				if (d.at<double>(0, 0) < dev)
				{
					dev = d.at<double>(0, 0);
					mean = m.at<double>(0, 0);
				}
			}
			in_out.at<uchar>(i, j) = mean;
		}
	}
}

void Scena::Wynik(int* t)
{
	if (wyniki.size() == 0){
		t[0] = t[1] = t[2] = t[3] = t[4] = t[5] = 0;
		return;
	}
	for (int i = 0; i < 6; ++i)
	{
		vector<Vec<int,2>> dom;
		for (int j = 0; j < wyniki.size(); j++)
		{
			bool jest = false;
			for (int k = 0; k < dom.size(); k++)
			{
				if (wyniki[j][i] == dom[k][0])
				{
					++dom[k][1];
					jest = true;
					break;
				}
			}
			if (!jest)
			{
				int temp[2];
				temp[0] = wyniki[j][i];
				temp[1] = 1;
				dom.emplace_back(temp);
			}
		}
		int ret = 0;
		for (int k = 1; k < dom.size(); k++)
		{
			if (dom[ret][1] < dom[k][1])
				ret = k;
		}
		t[i] = dom[ret][0];
	}
}

void Scena::maxCol(Mat in1,Mat in2, Mat &out){
	int row = in1.rows;
	int col = in1.cols;
	for (int i = 0; i < row; ++i){
		for (int j = 0; j < col; j++){
			uchar p1 = in1.at<uchar>(i, j);
			uchar p2 = in2.at<uchar>(i, j);
			out.at<uchar>(i, j) = p1 >= p2 ? p1 : p2;
		}
	}
}
vector<vector<Point>> Scena::wykryjKsztalty(Mat zdj)
{
	Mat temp, t;
	temp.create(Size(zdj.size()), CV_8UC1);
	minCol(zdj,temp,60); //TODO Sprawdzic dla mniejszego tresholda czy te¿ dzia³a
	medianBlur(temp, t, 7);
	Mat kernel(Size(4, 4), CV_8UC1);
	kernel.setTo(1);
	dilate(t, t, kernel);
	//Canny(t, temp,  100,300,3);
	vector<vector<Point>> contours;
	findContours(t, contours,CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
#ifdef Debug
	for (int i = 0; i< contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area<1000)
		{
			contours.erase(contours.begin() + i);
			--i;
		}
	}
	Mat drawing = Mat::zeros(t.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = (255,0,0);
		drawContours(drawing, contours, i, color, 2, 8,0, 0, Point());
	}
#endif
	/*for (int i = 0; i < contours.size(); i++)
	{
		convexHull(contours[i], contours[i]);
	}

#ifdef Debug
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = (0, 255, 0);
		drawContours(drawing, contours, i, color, 2, 8, 0, 0, Point());
	}
#endif*/
	return contours;
}

vector<Point> Scena::znajdzWierzcholki(vector<vector<Point>> kontury)
{
	for (int i = 0; i < kontury.size(); i++)
	{
		double area = contourArea(kontury[i]);
		if (area<3000 || area > 100000)
		{
			kontury.erase(kontury.begin() + i);
			if (kontury.size() < 4)
				return vector<Point>();
			--i;
		}
	}

#ifdef Debug
	Mat drawing = Mat::zeros(tempzd.size(), CV_8UC1);
	for (int i = 0; i< kontury.size(); i++)
	{
		drawContours(tempzd, kontury, i, CV_RGB(255, 0, 0), 2, 8, 0, 0, Point());
	}
#endif

	vector<vector<Point>>  approx;
	int Rec = 0, Cir = 0, Cor = 0;
	for (int i = 0; i < kontury.size(); ++i)
	{
		vector<Point> app;
		approxPolyDP(Mat(kontury[i]), app, 10, true);
		approx.emplace_back(app);
		if (app.size() == 4)
			++Rec;
		else if (app.size() == 6)
			++Cor;
		else
			++Cir;
	}
	vector<Point> pRec, pCir, pCor;
	double wRec = 0, wCir = 0, wCor = 0;
	if (Rec >= 4)
	{
		vector<vector<Point>> kon = redukujKontury(kontury, approx, 4, even);
		pRec = typPasy(kon);
		wRec = wskaznikProstokata(pRec, RECT);
	}
	else if (Cor >= 4)
	{
		//vector<vector<Point>> kon = redukujKontury(kontury, approx, 6, even);
		pCor = typRogi(approx);
		wCor = wskaznikProstokata(pCor, CORN);
	}

	pCir = typOkregi(kontury);
	wCir = wskaznikProstokata(pCir, CIRC);

#ifdef Debug
	for (int i = 0; i< approx.size(); i++)
	{
		for (int j = 0; j < approx[i].size(); ++j)
		{
			line(tempzd, approx[i][j], approx[i][j + 1 == approx[i].size() ? 0 : j + 1], CV_RGB(0, 255, 0));
		}
	}
#endif
	if (wCir>wRec && wCir>wCor)
	{
		znacznik = CIRC;
		return pCir;
	}
	else if (wRec > wCor)
	{
		znacznik = RECT;
		return pRec;
	}	
	else
	{
		znacznik = CORN;
		return pCor;
	}
}

void Scena::rzutuj(Mat &zdj, vector<Point>  wierzcholki)
{
	Point2f srcTri[4];
	Point2f dstTri[4];

	Mat rot_mat(2, 3, CV_32FC1);
	Mat warp_mat(2, 3, CV_32FC1);
	Mat warp_dst, warp_rotate_dst;

	if (znacznik == RECT)
		warp_dst = Mat::zeros((int)RECT_Y, (int)RECT_X, zdj.type());
	else if (znacznik==CIRC)
		warp_dst = Mat::zeros((int)CIRC_Y, (int)CIRC_X, zdj.type());
	else
		warp_dst = Mat::zeros((int)CORN_Y, (int)CORN_X, zdj.type());

	dstTri[0] = Point2f(0, 0);
	dstTri[1] = Point2f(warp_dst.cols - 1, 0);
	dstTri[2] = Point2f(0, warp_dst.rows - 1);
	dstTri[3] = Point2f(warp_dst.cols - 1, warp_dst.rows - 1);

	srcTri[0] = wierzcholki[0];
	srcTri[1] = wierzcholki[1];
	srcTri[2] = wierzcholki[3];
	srcTri[3] = wierzcholki[2];

	warp_mat = getPerspectiveTransform(srcTri, dstTri);
	warpPerspective(zdj, warp_dst, warp_mat, warp_dst.size());
	zdj = warp_dst.clone();
}
void Scena::minCol(Mat in, Mat out, int treshold){
	int row = in.rows;
	int col = in.cols;
	for (int i = 0; i < row; ++i){
		for (int j = 0; j < col; j++){
			Pixel p = in.at<Pixel>(i, j);
			uchar temp;
			if (p(0) > p(1) && p(0) > p(2))
				temp = p(0);
			else if (p(1) > p(2))
				temp = p(1);
			else
				temp = p(2);
			out.at<uchar>(i, j) = temp>=treshold ? 0:255;
		}
	}
}

vector<Point> Scena::typOkregi(vector<vector<Point>> kontury)
{	
	vector<Point> ret;
	
	for (int i = 0; i < kontury.size(); i++)
	{
		RotatedRect rec = minAreaRect(kontury[i]);
		
		
#ifdef Debug
		Point2f vertices[4];
		rec.points(vertices);
		for (int j = 0; j < 4; j++)
			line(tempzd, vertices[j], vertices[(j + 1) % 4], Scalar(0, 0, 255));
		double pole = contourArea(kontury[i]);
#endif

		int A = rec.size.width / 2;
		int B = rec.size.height / 2;
		double estimated_area = CV_PI*A*B;
		double err = fabs(estimated_area - contourArea(kontury[i])) / contourArea(kontury[i]);

		if (err>0.05)
		{
			kontury.erase(kontury.begin() + i);
			if (kontury.size() < 4)
				return ret;
			--i;
		}
	}
	for (int i = 0; i < kontury.size(); i++)
	{
		Point p;
		Moments m;
		m = moments(kontury[i]);
		p.x = m.m10 / m.m00;
		p.y = m.m01 / m.m00;
		ret.emplace_back(p);
#ifdef Debug
		circle(tempzd, p, 10, CV_RGB(0, 255, 255), 10);
#endif
	}
	return bestRect(ret, CIRC);
}
vector<Point> Scena::typPasy(vector<vector<Point>> kontury)
{
	vector<Point> ret;
	vector<vector<Point2f>> linie;
	for (size_t i = 0; i < kontury.size(); i++)
	{
		RotatedRect rec = minAreaRect(kontury[i]);
		double stosunek = rec.size.width<rec.size.height ? rec.size.width / rec.size.height : rec.size.height / rec.size.width;
		if (stosunek > 1 || stosunek < 1/10.0)
			continue;
		vector<Point2f> linia;
		Point2f p[4];
		rec.points(p);

#ifdef Debug
		for (int i = 0; i < 4; i++)
			line(tempzd, p[i], p[(i + 1) % 4], CV_RGB(0, 0, 255));
#endif

		if (pointsDistance(p[0], p[1])<pointsDistance(p[1], p[2]))
		{
			linia.emplace_back(Point2f((p[0].x + p[1].x) / 2, (p[0].y + p[1].y) / 2));
			linia.emplace_back(Point2f((p[2].x + p[3].x) / 2, (p[2].y + p[3].y) / 2));
		}
		else
		{
			linia.emplace_back(Point2f((p[2].x + p[1].x) / 2, (p[2].y + p[1].y) / 2));
			linia.emplace_back(Point2f((p[0].x + p[3].x) / 2, (p[0].y + p[3].y) / 2));
		}
		linie.emplace_back(linia);
#ifdef Debug
		line(tempzd, linia[0], linia[1],CV_RGB(255,255,0));
#endif
	}

	if (linie.size() < 4 ||linie.size() > 7)
		return vector<Point>{};

	vector<Point> tempRect;
	double wsk=0;
	for (size_t i1 = 0; i1 <= linie.size() - 4; i1++)
	{
		for (size_t i2 = i1; i2 <= linie.size() - 4; i2++)
		{
			for (size_t i3 = i2; i3 <= linie.size() - 4; i3++)
			{
				for (size_t i4 = i3; i4 <= linie.size() - 4; i4++)
				{
					vector<vector<Point2f>> temp{ linie[i1], linie[i2 + 1], linie[i3 + 2], linie[i4 + 3] };

					for (size_t i = 0; i < 3; i++)
					{
						for (size_t j = i + 1; j < 4; j++)
						{
							Point p = punktPrzeciecia(temp[i], temp[j]);
							if (p.x>0 && p.y>0 && p.x < rozmiarObrazu.width && p.y << rozmiarObrazu.height)
							{
								tempRect.emplace_back(p);
#ifdef Debug
								circle(tempzd, punktPrzeciecia(linie[i], linie[j]), 10, CV_RGB(0, 255, 255), 10);
#endif
							}
						}
					}
					if (tempRect.size() == 4){
						rectOrder(tempRect);
						double tt = wskaznikProstokata(tempRect, RECT);
						if (tt > wsk)
						{
							wsk = tt;
							ret = tempRect;
						}
					}
					tempRect.clear();
				}
			}
		}
	}
	return ret;
}
vector<Point> Scena::typRogi(vector<vector<Point>> approx)
{
	vector<Point> ret;
	for (int i = 0; i < approx.size(); i++)
	{
		if (approx[i].size() != 6)
		{
			approx.erase(approx.begin() + i);
			--i;
			continue;
			if (approx.size() < 4)
				return vector<Point>();
		}
		ret.emplace_back(approx[i][findCenter(approx[i])]);
#ifdef Debug
		circle(tempzd, ret[i], 10, CV_RGB(0, 255, 255), 10);
#endif
	}
	if (ret.size() > 4)
	{
		for (size_t i = 0; i < ret.size(); i++)
		{
			if (!jestRogiem(approx[i],ret[i]))
			{
				approx.erase(approx.begin() + i);
				ret.erase(ret.begin() + i);
				--i;
				continue;

			}
		}
	}
	return bestRect(ret, CORN);
}
vector<vector<Point>> Scena::redukujKontury(vector<vector<Point>> kontury, vector<vector<Point>> approx, int b, TRYB t)
{
	if (kontury.size() == 4)
		return kontury;
	for (int i = approx.size()-1; i >=0; --i)
	{
		if (t == greater){
			if (approx[i].size() <= b)
				kontury.erase(kontury.begin() + i);
		}
		else
		{
			if (approx[i].size() < b || approx[i].size() > b)
				kontury.erase(kontury.begin() + i);
		}
	}
	return kontury;
}
Point Scena::punktPrzeciecia(vector<Point2f> l1, vector<Point2f> l2)
{
	double A, B, D;
	A = l1[0].x*l1[1].y - l1[0].y*l1[1].x;
	B = l2[0].x*l2[1].y - l2[0].y*l2[1].x;
	D = (l1[0].x - l1[1].x)*(l2[0].y - l2[1].y) - (l1[0].y - l1[1].y)*(l2[0].x - l2[1].x);
	if (D == 0)
		return Point(-1, -1);
	int x = (A*(l2[0].x - l2[1].x)-B*(l1[0].x-l1[1].x))/D;
	int y = (A*(l2[0].y - l2[1].y) - B*(l1[0].y - l1[1].y))/D;
	return Point(x, y);
}
int Scena::findCenter(vector<Point> p)
{
	Point center;
	double x=0, y=0;
	for (size_t i = 0; i < p.size(); i++)
	{
		x += p[i].x;
		y += p[i].y;
	}
	center.x = x / p.size();
	center.y = y / p.size();
	int ret=0;
	double dist = pointsDistance(center,p[ret]);
	for (size_t i = 1; i < p.size(); i++)
	{
		double temp = pointsDistance(p[i], center);
		if (dist>temp)
		{
			dist = temp;
			ret = i;
		}
	}
	return ret;
}
bool Scena::jestRogiem(vector<Point>ksztalt,Point center)
{
	size_t i;
	for (i = 0; i < ksztalt.size(); i++)
	{
		if (ksztalt[i] == center)
			break;
	}
	if (i == 6)
		return false;
	for (size_t j = 0; j < 2; j++)
	{
		int a1 = (i + j) % 6;
		int a2 = (a1+1) % 6;

		int b1 = (a2+1) % 6;
		int b2 = (b1+1) % 6;

		int c1 = (b2+1) % 6;
		int c2 = (c1+1) % 6;
		float err = abs(angleBetween(ksztalt[a1]-ksztalt[a2],ksztalt[b2]-ksztalt[b1]));
#ifdef Debug
		Scalar col = CV_RGB(rand() % 256, rand() % 256, rand() % 256);
		line(tempzd,ksztalt[a1],ksztalt[a2],col,3);
		line(tempzd, ksztalt[b1], ksztalt[b2], col, 3);
#endif
		if (err>0.3)
			return false;

		err = abs(angleBetween(ksztalt[a1] - ksztalt[a2], ksztalt[c1] - ksztalt[c2]));
#ifdef Debug
		col = CV_RGB(rand() % 256, rand() % 256, rand() % 256);
		line(tempzd, ksztalt[a1], ksztalt[a2], col, 3);
		line(tempzd, ksztalt[c1], ksztalt[c2], col, 3);
#endif
		if (err>0.3)
			return false;
	}
	return true;
}
double pointsDistance(Point2f p1, Point2f p2){
	return abs(sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y)));
}
float Scena::angleBetween(const Point &v1, const Point &v2)
{
	float len1 = sqrt(v1.x * v1.x + v1.y * v1.y);
	float len2 = sqrt(v2.x * v2.x + v2.y * v2.y);

	float dot = v1.x * v2.x + v1.y * v2.y;

	float a = dot / (len1 * len2);

	if (a >= 1.0)
		return 0.0;
	else if (a <= -1.0)
		return CV_PI;
	else
		return acos(a); // 0..PI
}
vector<Point> Scena::bestRect(vector<Point> p, ZNACZNIK znacznik)
{
#ifdef Debug
	if (p.size() > 4)
	{
		int debug = -1;
	}
#endif
	vector<Point> ret;
	if (p.size()<4 || p.size()>7)
		return ret;
	double wsk=0;
	for (size_t i1 = 0; i1 <= p.size()-4; i1++)
	{
		for (size_t i2 = i1; i2 <= p.size() - 4; i2++)
		{
			for (size_t i3 = i2; i3 <= p.size() - 4; i3++)
			{
				for (size_t i4 = i3; i4 <= p.size() - 4; i4++)
				{
					vector<Point> temp{p[i1], p[i2 + 1], p[i3 + 2], p[i4 + 3] };
					rectOrder(temp);
					double twsk = wskaznikProstokata(temp,znacznik);
					if (twsk > wsk)
					{
						wsk = twsk;
						ret = temp;
					}
				}
			}
		}
	}
	return ret;
}
double Scena::wskaznikProstokata(vector<Point> rec, ZNACZNIK znacznik)
{
	if (rec.empty())
		return 0;
	double ret = CV_PI;
	double stosunek = (pointsDistance(rec[0], rec[1]) + pointsDistance(rec[2], rec[3])) 
					/ (pointsDistance(rec[0], rec[3]) + pointsDistance(rec[1], rec[2]));
	ret -= (angleBetween(rec[0] - rec[1], rec[3] - rec[2]));
	ret -= (angleBetween(rec[0] - rec[3], rec[1] - rec[2]));
	return ret * (1 - abs(stosunek - znacznik));
}
void Scena::rectOrder(vector<Point> &rec)
{
#ifdef Debug
	for (size_t i = 0; i < 4; i++)
		putText(tempzd, to_string(i), rec[i], 5, 10, Scalar(100, 255, 100), 2);
#endif
	int a=0, b, c, d;
	for (size_t i = 1; i < rec.size(); i++)
	{
		if (pointsDistance(Point(0, 0), rec[a]) > pointsDistance(Point(0, 0), rec[i]))
			a = i;
	}
	int temp = a == 0 ? rec[1].y : rec[0].y;
	for (int i = 0; i < rec.size(); i++)
	{
		if (i == a)
			continue;
		if (rec[i].y <= temp)
		{
			temp = rec[i].y;
			b = i;
		}
	}
	temp = 0;
	for (size_t i = 0; i < rec.size() - 1; i++)
	{
		if (i == a || i == b)
			continue;
		if (rec[i].x>temp)
		{
			temp = rec[i].x;
			c = i;
		}
	}

	for (d = 0; d < rec.size(); d++)
	{
		if (d == a || d==b || d==c)
			continue;
		else
			break;
	}
	if (pointsDistance(rec[a], rec[b])>pointsDistance(rec[a], rec[d]))
		rec = { rec[a], rec[b], rec[c], rec[d] };
	else
		rec = { rec[a], rec[d], rec[c], rec[b] };
#ifdef Debug
	for (size_t i = 0; i < 4; i++)
		putText(tempzd, to_string(i), rec[i], 4, 10, Scalar(255, 255,100),4);
#endif
}

void maxCol(Mat in1, Mat in2, Mat &out){
	int row = in1.rows;
	int col = in1.cols;
	for (int i = 0; i < row; ++i){
		for (int j = 0; j < col; j++){
			if (in1.at<uchar>(i, j) == 0 || in2.at<uchar>(i, j) == 0)
				out.at<uchar>(i, j) = 0;
			else
				out.at<uchar>(i, j) = 255;
		}
	}
}

void SimplestCB(Mat& in, Mat& out, float percent) {
	assert(in.channels() == 3);
	assert(percent > 0 && percent < 100);

	float half_percent = percent / 200.0f;

	vector<Mat> tmpsplit; split(in, tmpsplit);
	for (int i = 0; i<3; i++) {
		Mat flat; tmpsplit[i].reshape(1, 1).copyTo(flat);
		cv::sort(flat, flat, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
		int lowval = flat.at<uchar>(cvFloor(((float)flat.cols) * half_percent));
		int highval = flat.at<uchar>(cvCeil(((float)flat.cols) * (1.0 - half_percent)));

		tmpsplit[i].setTo(lowval, tmpsplit[i] < lowval);
		tmpsplit[i].setTo(highval, tmpsplit[i] > highval);

		normalize(tmpsplit[i], tmpsplit[i], 0, 255, NORM_MINMAX);
	}
	merge(tmpsplit, out);
}

void normalizuj(vector<Point>p, Mat &matNormalizacji, ZNACZNIK z)
{
	float x1, x2, y1, y2;
	if (z == CIRC)
	{
		x1 = CIRC_X / pointsDistance(p[0], p[1]);
		x2 = CIRC_X / pointsDistance(p[2], p[3]);
		y1 = CIRC_Y / pointsDistance(p[0], p[3]);
		y2 = CIRC_Y / pointsDistance(p[2], p[1]);
	}
	else if (z == RECT)
	{
		x1 = RECT_X / pointsDistance(p[0], p[1]);
		x2 = RECT_X / pointsDistance(p[2], p[3]);
		y1 = RECT_Y / pointsDistance(p[0], p[3]);
		y2 = RECT_Y / pointsDistance(p[2], p[1]);
	}
	else
	{
		x1 = CORN_X / pointsDistance(p[0], p[1]);
		x2 = CORN_X / pointsDistance(p[2], p[3]);
		y1 = CORN_Y / pointsDistance(p[0], p[3]);
		y2 = CORN_Y / pointsDistance(p[2], p[1]);
	}
	
	int row = matNormalizacji.rows;
	int col = matNormalizacji.cols;
	for (int i = 0; i < row; ++i){
		for (int j = 0; j < col; j++){
			float temp,temp2,a,b;
			Point tp(i, j);
			temp = minimum_distance(p[0], p[1], tp);
			if (temp == 0)
				a = x1;
			else {
				temp2 = minimum_distance(p[2], p[3], tp);
				if (temp2 == 0)
					a = x2;
				else
					a =(temp*x2+temp2*x1) / (temp2+temp);
			}
			temp = minimum_distance(p[0], p[3], tp);
			if (temp == 0)
				b = y1;
			else {
				temp2 = minimum_distance(p[2], p[1], tp);
				if (temp2 == 0)
					b = y2;
				else
					b = (temp*y2 + temp2*y1) / (temp2 + temp);
			}

			matNormalizacji.at<float>(i, j)=(a+b)/2;
		}
	}
}
float minimum_distance(Point v, Point w, Point p)
{
	const float l2 = pointsDistance(v, w)* pointsDistance(v, w);  
	if (l2 == 0.0) return pointsDistance(p, v);   
	const float t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2;
	if (t < 0.0) return pointsDistance(p, v);       
	else if (t > 1.0) return pointsDistance(p, w);  
	return pointsDistance(p, Point(v.x + t * (w.x - v.x), v.y + t * (w.y - v.y)));
}

void Scena::policzZelki(Mat &zdj, vector<Point> p)
{
	Mat hsv, lab,out1,out2;
	int wynik[6];
	cvtColor(zdj, hsv, CV_BGR2HSV_FULL);
	cvtColor(zdj, lab, CV_BGR2Lab);
	//zielone
	inRange(hsv, Scalar(32, 49, 11), Scalar(55, 255, 161), out1);
	//inRange(hsv, Scalar(38, 109, 13), Scalar(109, 255, 154), out1);
	policzKolorowe(hsv, out1, p, wynik[2]);
	//pomarañczowe
	//inRange(hsv, Scalar(8,175,175), Scalar(20,255,255), out1);
	inRange(hsv, Scalar(7, 172, 52), Scalar(21, 255, 255), out1);
	policzKolorowe(hsv, out1, p, wynik[3]);
	//¿ó³te
	//inRange(hsv, Scalar(21,156,146), Scalar(70,255,255), out1);
	inRange(hsv, Scalar(17, 156, 146), Scalar(70, 255, 255), out1);
	policzKolorowe(hsv, out1, p, wynik[5]);
	//czerwone
	inRange(hsv, Scalar(238, 45,120), Scalar(255, 255,255), out1);
	inRange(hsv, Scalar(0, 95,100), Scalar(10, 255, 183), out2);
	add(out1, out2, out1);
	policzKolorowe(hsv, out1, p, wynik[0]);
	//jasne czerwone
	//inRange(hsv, Scalar(250, 173, 151), Scalar(255, 255, 255), out1);
	//inRange(hsv, Scalar(0, 173, 151), Scalar(5, 255, 255), out2);
	inRange(hsv, Scalar(243, 90, 16), Scalar(255, 255, 137), out1);
	//add(out1, out2, out1);
	policzKolorowe(hsv, out1, p, wynik[1]);
	//bia³e
	//inRange(hsv, Scalar(21, 53, 139), Scalar(57, 122, 255), out1);
	inRange(hsv, Scalar(17, 56, 124), Scalar(151, 147, 255), out1);
	//maxCol(out1, out2, out1);
	policzPrzezroczyste(hsv, out1, p, wynik[4]);

	wyniki.emplace_back(wynik);
}
void Scena::policzKolorowe(Mat &zdj, Mat &binary, vector<Point> p, int &liczba)
{
	liczba = 0;
	Mat kernel;
	kernel.create(Size(3,3), CV_8UC1);
	kernel.setTo(1);
	erode(binary, binary, kernel);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
#ifdef Debug
	Mat drawing;
	drawing.create(zdj.size(), CV_8UC3);
	Mat bin;
	bin = binary.clone();
#endif
	findContours(binary, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	int next = 0;
	if (contours.size() == 0)
		return;
	while (next != -1)
	{
		vector<Point>  obrys;
		approxPolyDP(contours[next], obrys, 5, true);
		RotatedRect rec = minAreaRect(obrys);
		Point2f po[4];
		rec.points(po);
		float width;
		float height;
		if (rec.size.width < rec.size.height){
			width = rec.size.width;
			height = rec.size.height;
		}
		else
		{
			width = rec.size.height;
			height = rec.size.width;
		}

		if (width < 100 || height < 200)
		{
			next = hierarchy[next][0];
			continue;
		}
		if (width>250 || height>350)
		{
			liczba += kupaZelkow(zdj,p,next, contours, hierarchy);
			next = hierarchy[next][0];
			continue;
		}
		
		float tem = width / height;
		double pole = contourArea(contours[next]);
		float stosunek = pole / (width*height);
		if (stosunek < 0.5 || (tem<0.5 && pole < 16000)){
			next = hierarchy[next][0];
			continue;
		}


		liczba += 1;
		fillConvexPoly(zdj, contours[next], Scalar(0, 0, 0));
		fillConvexPoly(zdj, obrys, Scalar(0, 0, 0));

#ifdef Debug	
		Scalar color = Scalar(255, 0, 0);
		for (int i = 0; i< obrys.size()-1; i++)
			line(drawing, obrys[i], obrys[i+1], color);
		line(drawing, obrys[0], obrys[obrys.size()-1], color);
		Point2f p[4];
		rec.points(p);
		line(drawing, p[0], p[1], Scalar(0, 0, 255));
		line(drawing, p[2], p[1], Scalar(0, 0, 255));
		line(drawing, p[2], p[3], Scalar(0, 0, 255));
		line(drawing, p[0], p[3], Scalar(0, 0, 255));
#endif
		next = hierarchy[next][0];

	}

#ifdef Debug
	RNG rng(12345);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
	}
#endif
	
}
void Scena::policzPrzezroczyste(Mat &zdj, Mat &binary, vector<Point> p, int &liczba)
{
	liczba = 0;
	Mat kernel;
	kernel.create(Size(10, 10), CV_8UC1);
	kernel.setTo(1);
	erode(binary, binary, kernel);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
#ifdef Debug
	Mat drawing;
	drawing.create(zdj.size(), CV_8UC3);
	Mat bin;
	bin = binary.clone();
#endif
	findContours(binary, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	int next = 0;
	if (contours.size() == 0)
		return;
	while (next != -1)
	{
		vector<Point>  obrys;
		approxPolyDP(contours[next], obrys, 5, true);
		RotatedRect rec = minAreaRect(contours[next]);
		Point2f po[4];
		rec.points(po);
		float width;
		float height;
		if (rec.size.width < rec.size.height){
			width = rec.size.width;
			height = rec.size.height;
		}
		else
		{
			width = rec.size.height;
			height = rec.size.width;
		}

		if (width < 100 || height < 200)
		{
			next = hierarchy[next][0];
			continue;
		}
		if (width>250 || height>350)
		{
			liczba += kupaZelkowbial(zdj, p, next, contours, hierarchy);
			next = hierarchy[next][0];
			continue;
		}

		float tem = width / height;
		double pole = contourArea(contours[next]);
		int tnext = hierarchy[next][2];
		while (tnext != -1)
		{
			pole -= contourArea(contours[tnext]);
			tnext = hierarchy[tnext][0];
		}
		float stosunek = pole / (width*height);
		if (stosunek < 0.60 || (tem<0.5 && pole < 16000)){
			next = hierarchy[next][0];
			continue;
		}


		liczba += 1;
#ifdef Debug
		fillConvexPoly(zdj, contours[next], Scalar(0, 0, 0));
	
		Scalar color = Scalar(255, 0, 0);
		for (int i = 0; i< obrys.size() - 1; i++)
			line(drawing, obrys[i], obrys[i + 1], color);
		line(drawing, obrys[0], obrys[obrys.size() - 1], color);
		Point2f p[4];
		rec.points(p);
		line(drawing, p[0], p[1], Scalar(0, 0, 255));
		line(drawing, p[2], p[1], Scalar(0, 0, 255));
		line(drawing, p[2], p[3], Scalar(0, 0, 255));
		line(drawing, p[0], p[3], Scalar(0, 0, 255));
#endif
		next = hierarchy[next][0];

	}

#ifdef Debug
	RNG rng(12345);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
	}
#endif

}
int Scena::kupaZelkow(Mat &zdj, vector<Point> p, int nr_konturu, vector<vector<Point> > contours, vector<Vec4i> hierarchy)
{
	Mat mPol;
	mPol.create(zdj.size(), CV_8UC1);
	mPol.setTo(0);
	Mat obrys;
	approxPolyDP(contours[nr_konturu], obrys, 5, true);
	int next = hierarchy[nr_konturu][2];
	fillConvexPoly(mPol, obrys, 255);

	double pole = contourArea(obrys);
	while (next != -1)
	{
		fillConvexPoly(mPol, contours[next], 0);
		pole -= contourArea(contours[next]);
		next = hierarchy[next][0];
	}

	RotatedRect rec = minAreaRect(obrys);
	Point2f po[4];
	rec.points(po);
	float width;
	float height;
	if (rec.size.width < rec.size.height){
		width = rec.size.width;
		height = rec.size.height;
	}
	else
	{
		width = rec.size.height;
		height = rec.size.width;
	}
	float stosunek = pole / (width*height);
	cvtColor(mPol, mPol, CV_GRAY2RGB);
	subtract(zdj, mPol, zdj);
	int ret=round(pole / 32000.0);
	return ret == 0 ? 1 : ret;
}

int Scena::kupaZelkowbial(Mat &zdj, vector<Point> p, int nr_konturu, vector<vector<Point> > contours, vector<Vec4i> hierarchy)
{
	Mat mPol;
	mPol.create(zdj.size(), CV_8UC1);
	mPol.setTo(0);
	Mat obrys;
	approxPolyDP(contours[nr_konturu], obrys, 5, true);
	int next = hierarchy[nr_konturu][2];
	fillConvexPoly(mPol, contours[nr_konturu], 255);
	double pole = contourArea(contours[nr_konturu]);
	while (next != -1)
	{
		fillConvexPoly(mPol, contours[next], 0);
		pole -= contourArea(contours[next]);
		next = hierarchy[next][0];
	}

	RotatedRect rec = minAreaRect(obrys);
	Point2f po[4];
	rec.points(po);
	float width;
	float height;
	if (rec.size.width < rec.size.height){
		width = rec.size.width;
		height = rec.size.height;
	}
	else
	{
		width = rec.size.height;
		height = rec.size.width;
	}
	
	float stosunek = pole / (width*height);
	if (pole < 30000 && stosunek < 0.6)
		return 0;
	cvtColor(mPol, mPol, CV_GRAY2RGB);
	subtract(zdj, mPol, zdj);
	int ret = round(pole / 34000.0);
	return ret == 0 ? 1 : ret;
}