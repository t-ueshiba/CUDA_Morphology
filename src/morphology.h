#ifndef MORPHOLOGY_H
#define MORPHOLOGY_H

#include <cuda_runtime.h>


/*!
 * \file morphology.h
 *
 * We use the van Herk/Gil-Werman (vHGW) algorithm, [van Herk,
 * Patt. Recog. Let. 13, pp. 517-521, 1992; Gil and Werman,
 * IEEE Trans PAMI 15(5), pp. 504-507, 1993.]
 *
 * Please refer to the leptonica documents for more details:
 * http://www.leptonica.com/binary-morphology.html
 *
 */
inline int	roundUp(int x, int y)	{ return (x + y - 1) / y; }


/*!
 * \brief erode()
 *
 * \param[in/out]   img_d: device memory pointer to source image
 * \param[in]       width: image width
 * \param[in]       height: image height
 * \param[in]       hsize: horizontal size of Sel; must be odd; origin implicitly in center
 * \param[in]       vsize: ditto
 */
extern "C" void	erode(unsigned char* img_d,
		      int width, int height, int hsize, int vsize);


/*!
 * \brief dilate()
 *
 * \param[in/out]   img_d: device memory pointer to source image
 * \param[in]       width: image width
 * \param[in]       height: image height
 * \param[in]       hsize: horizontal size of Sel; must be odd; origin implicitly in center
 * \param[in]       vsize: ditto
 */
extern "C" void	dilate(unsigned char* img_d,
		       int width, int height, int hsize, int vsize);

#endif /* MORPHOLOGY_H */
