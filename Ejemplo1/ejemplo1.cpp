//#include <iostream>
#include <cstdio>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

int main() {
	// Nombre de la imagen a cargar
	/*char nombreImagen[] = "C:\\Users\\Edwin\\Documents\\Desarrollos\\CursoOpenCV\\Escritura_primer_programa\\ivvi.jpg";
	Mat imagen;
	//Mat dst;
	//Mat imagenResultadoLut;
	Mat hsvimagen;
	//int i, j;
	imagen = cv::imread(nombreImagen);
	if (!imagen.data){
		cout << "No se puede cargar la imagen:[" << nombreImagen << "]" << endl;
		exit(1);
	}*/


	/*for (i = imagen.rows / 4; i < 3 * imagen.rows / 4; ++i) {
		for (j = imagen.cols / 4; j < 3 * imagen.cols / 4; ++j) {
			imagen.at<Vec3b>(Point(i, j)) = Vec3b(0, 0, 0);
		}
	}

	namedWindow("ImagenPrueba", WINDOW_AUTOSIZE);
	imshow("ImagenPrueba", imagen);

	// Guardo el resultado
	imwrite("Resultado.jpg", imagen);
	*/

	/// Separar la imagen a 3 subimagenes ( A, V y R )
	/*vector<Mat> bgr_planes;
	split(imagen, bgr_planes);

	//Variables para el histograma
	int histSize = 256;

	/// los rangos (A,V,R) 
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true;
	bool accumulate = false;

	Mat b_hist, g_hist, r_hist;

	//calcular el histograma
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	cout << "hist: " << b_hist << endl;

	// Dibujar el histograms para A, V y R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalizar el resultado a [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	cout << "hist: " << b_hist << endl;

	/// Dibujar para  cada canal
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	//Mostrar la imagen
	namedWindow("Original", CV_WINDOW_AUTOSIZE);
	imshow("Original", imagen);
	namedWindow("Histograma", CV_WINDOW_AUTOSIZE);
	imshow("Histograma", histImage);
	*/
	/// Ejercicio LUT
	/*Mat lut(1, 256, CV_8U);

	for (int i = 0; i < 256; i++) {
		//lut.at<uchar>(i) = 255 - i; //Función Inversa
		lut.at<char>(1) = pow((float)i * 255, (float)(1 / 2.0));
	}
	LUT(imagen, lut, imagenResultadoLut);

	//Mostrar la imagen
	namedWindow("Original", CV_WINDOW_AUTOSIZE);
	imshow("Original", imagen);

	namedWindow("ImagenResultadoLUT", CV_WINDOW_AUTOSIZE);
	imshow("ImagenResultadoLUT", imagenResultadoLut);

	//Esperar a pulsar una tecla
	cvWaitKey(0);
	return 0;*/

	/// Ejercicio de espacio de color
	//Separar la imagen a 3 subimagenes ( A, V y R )
	/*vector<Mat> bgr_planes;
	split(imagen, bgr_planes);

	cvtColor(imagen, hsvimagen, CV_BGR2HSV);
	vector<Mat> hsv_planes;
	split(hsvimagen, hsv_planes);

	//Mostrar la imagen

	namedWindow("Original", CV_WINDOW_AUTOSIZE);
	imshow("Original", imagen);

	namedWindow("HSV", CV_WINDOW_AUTOSIZE);
	imshow("HSV", hsvimagen);*/


	/// Ejercicio de operaciones
	//char NombreImagen1[] = "C:\\Users\\Edwin\\Documents\\Desarrollos\\CursoOpenCV\\24 Operaciones matemáticas con las OpenCV\\LSI.jpg";
	//char NombreImagen2[] = "C:\\Users\\Edwin\\Documents\\Desarrollos\\CursoOpenCV\\24 Operaciones matemáticas con las OpenCV\\UC3M.jpg";
	//Mat img1, img2;

	
	//Cargamos la imagen y se comprueba que lo ha hecho correctamente
	//img1 = imread(NombreImagen1);
	//img2 = imread(NombreImagen2);

	/*if (!img1.data || !img2.data) {
		cout << "Error al cargar las imagenes" << endl;
		exit(1);
	}
	imshow("Image1", img1);
	imshow("Image2", img2);

	Mat dst;*/

	//add(img1, img2, dst, noArray(), -1);
	//imshow("ADD", dst);
	//subtract(img1, img2, dst, noArray(), -1);
	//imshow("SUBTRACT", dst);
	//multiply(img1, img2, dst, (1.0), -1);
	//imshow("MULTIPLY", dst);
	//divide(img1, img2, dst, (1.0), -1);
	//imshow("DIVIDE", dst);

	//Mat res;

	//bitwise_and(img1, img2, res);
	//imshow("AND", res);
	//bitwise_or(img1, img2, res);
	//imshow("OR", res);
	//bitwise_not(img1, res);
	//imshow("NOT", res);
	/*bitwise_xor(img1, img2, res);
	imshow("XOR", res);*/

	/// Ejercicio de convolucion
	/*char NombreImagen[] = "C:\\Users\\Edwin\\Documents\\Desarrollos\\CursoOpenCV\\29 Ejemplo de Conv. de imágenes digitales con las OpenCV\\IMG.jpg";
	Mat src, dst;

	src = imread(NombreImagen);
	if (!src.data) {
		cout << "Error al cargar la imagen" << endl;
		exit(1);
	}
	Mat kernel;
	kernel = Mat::ones(5, 5, CV_32F) / (float)(5 * 5);
	cout << "kernel: " << kernel << endl;

	filter2D(src, dst, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
	
	namedWindow("src", CV_WINDOW_AUTOSIZE);
	imshow("src", src);
	namedWindow("dst", CV_WINDOW_AUTOSIZE);
	imshow("dst", dst);*/
	
	/// Ejercicio de correlación
	/*char NombreImagen[] = "C:\\Users\\Edwin\\Documents\\Desarrollos\\CursoOpenCV\\32 Correlación de imágenes digitales con las OpenCV\\corr_norm.tif";
	char Patron[] = "C:\\Users\\Edwin\\Documents\\Desarrollos\\CursoOpenCV\\32 Correlación de imágenes digitales con las OpenCV\\modelo.tif";

	Mat src;
	Mat temp;

	src = imread(NombreImagen);
	temp = imread(Patron);

	if (!src.data || !temp.data) {
		cout << "No se puede cargar la informacion" << endl;
		exit(1);
	}

	Mat ftmp[6];
	int iwidth;
	int iheight;

	// reservo la memora dependiendo del tamaño
	iwidth = src.cols - temp.cols + 1;
	iheight = src.rows - temp.rows + 1;

	for (int i = 0; i < 6; i++) {
		ftmp[i].create(iheight, iwidth, CV_32SC1);
	}

	// Correlacion
	for (int i = 0; i < 6; i++) {
		matchTemplate(src, temp, ftmp[i], i);
			normalize(ftmp[i], ftmp[i], 1, 0, CV_MINMAX);
	}

	namedWindow("Modelo", CV_WINDOW_AUTOSIZE);
	imshow("Modelo", temp);
	namedWindow("Imagen", CV_WINDOW_AUTOSIZE);
	imshow("Imagen", src);

	
	// Salidas
	namedWindow("SQDIFF", CV_WINDOW_AUTOSIZE);
	imshow("SQDIFF", ftmp[0]);

	namedWindow("SQDIFF_NORMED", CV_WINDOW_AUTOSIZE);
	imshow("SQDIFF_NORMED", ftmp[1]);

	namedWindow("CCORR", CV_WINDOW_AUTOSIZE);
	imshow("CCORR", ftmp[2]);

	namedWindow("CCORR_NORMED", CV_WINDOW_AUTOSIZE);
	imshow("CCORR_NORMED", ftmp[3]);

	namedWindow("CCOEFF", CV_WINDOW_AUTOSIZE);
	imshow("CCOEFF", ftmp[4]);

	namedWindow("CCOEFF_NORMED", CV_WINDOW_AUTOSIZE);
	imshow("CCOEFF_NORMED", ftmp[5]);*/

	/// Ejercicio 2 de Correlacion
	//Nombre de la imagen que se va a cargar
	/*char NombreImagen[] = "C:\\Users\\Edwin\\Documents\\Desarrollos\\CursoOpenCV\\33 Ejemplo de Correlación de imágenes digitales con las OpenCV\\IMG.jpg";
	char NombreModelo[] = "C:\\Users\\Edwin\\Documents\\Desarrollos\\CursoOpenCV\\33 Ejemplo de Correlación de imágenes digitales con las OpenCV\\MJ.jpg";

	//Cargamos las imagenes y se comprueba que lo ha hecho correctamente
	Mat src = imread(NombreImagen);
	Mat templ = imread(NombreModelo);
	
	if (!src.data || !templ.data) {
		cout << "Error al cargar la imagenes" << endl;
		exit(1);
	}

	Mat dst;

	//Reservamos memoria para los diversos metodos
	int iwidth = src.cols - templ.cols + 1;
	int iheight = src.rows - templ.rows + 1;

	dst.create(iheight, iwidth, CV_32FC1);

	int match_method = CV_TM_CCOEFF_NORMED;

	//Correlacion	
	matchTemplate(src, templ, dst, match_method);
	normalize(dst, dst, 0, 1, NORM_MINMAX, -1, Mat());

	double minVal, maxMal;
	Point minLoc, maxLoc;
	Point matchLoc;

	minMaxLoc(dst, &minVal, &maxMal, &minLoc, &maxLoc, Mat());

	//cout << "Min:" << minVal << "Max: " << maxMal << endl;

	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}

	rectangle(src, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(255, 0, 0), 4, 8, 0);
	rectangle(dst, Point(matchLoc.x - (templ.cols / 2), matchLoc.y - (templ.rows / 2)), Point(matchLoc.x + (templ.cols / 2), matchLoc.y + (templ.rows / 2)), Scalar::all(0), 4, 8, 0);

	imshow("src", src);
	imshow("Result", dst);
	imshow("templ", templ);*/

	/// Ejercicio trasnformaciones geometrica
	/*char NombreImagen[] = "C:\\Users\\Edwin\\Documents\\Desarrollos\\CursoOpenCV\\37 Ejemplo de Transformaciones geometricas con las OpenCV\\IVVI.jpg";
	Mat src, dst;

	//Cargamos la imagen y se comprueba que lo ha hecho correctamente
	src = imread(NombreImagen);
	if (!src.data) {
		cout << "Error al cargar la imagen: " << NombreImagen << endl;
		exit(1);
	}
	imshow("src", src);

	//Scale
	resize(src, dst, Size(0, 0), 0.5, 0.5, INTER_LINEAR);
	imshow("scale", dst);

	//Translation
	dst = Mat::zeros(src.size(), src.type());
	src(Rect(100, 100, src.cols - 100, src.rows - 100)).copyTo(dst(cv::Rect(0, 0, src.cols - 100, src.rows - 100)));
	imshow("translate", dst);

	//Rotate
	int len = max(src.cols, src.rows);
	double angle = 60;
	Point2f pt(len / 2.0, len / 2.0);
	Mat r = getRotationMatrix2D(pt, angle, 1.0);
	warpAffine(src, dst, r, Size(len, len));
	imshow("rotate", dst);

	//Reflection
	flip(src, dst, 1);
	imshow("reflection", dst);*/

	/// Reduccion de ruido
	char NombreImagen[] = "C:\\Users\\Edwin\\Documents\\Desarrollos\\CursoOpenCV\\40 Reducción de ruido con las OpenCV\\ivvi_512x512_gray_rg.jpg";
	char NombreImagenivi[] = "C:\\Users\\Edwin\\Documents\\Desarrollos\\CursoOpenCV\\40 Reducción de ruido con las OpenCV\\ivvi_512x512_gray.jpg";
	Mat src, imBlr, imGus, imMed, imBil, ivvi;

	//Cargamos la imagen y se comprueba que lo ha hecho correctamente
	src = imread(NombreImagen);
	ivvi = imread(NombreImagenivi);
	if (!src.data) {
		cout << "Error al cargar la imagen: " << NombreImagen << endl;
		exit(1);
	}
	// Homogeneous blur
	blur(src, imBlr, Size(5, 5), Point(-1, -1), 4);

	// Gaussian blur
	GaussianBlur(src, imGus, Size(5, 5), 0, 0, 4);

	// Median blur
	medianBlur(src, imMed, 5);

	//bilateral filter
	bilateralFilter(src, imBil, 15, 80, 80);

	//Mostrar la imagenes
	namedWindow("Vehiculo IVVI", CV_WINDOW_AUTOSIZE);
	imshow("Vehiculo IVVI", ivvi);

	namedWindow("Original con ruido", CV_WINDOW_AUTOSIZE);
	imshow("Original con ruido", src);

	namedWindow("blur", CV_WINDOW_AUTOSIZE);
	imshow("blur", imBlr);

	namedWindow("Gaussian", CV_WINDOW_AUTOSIZE);
	imshow("Gaussian", imGus);

	namedWindow("Median", CV_WINDOW_AUTOSIZE);
	imshow("Median", imMed);

	namedWindow("bilateral", CV_WINDOW_AUTOSIZE);
	imshow("bilateral", imBil);





	//Esperar a pulsar una tecla
	cvWaitKey(0);
	return 0;
}