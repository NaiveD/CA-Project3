////////////////
// 
// File: kmeans.cpp
//
//  Main body of K-Means simulaton. Reads in the original data points from
//  `ori.txt`, performs K-Means clustering on randomly-picked initial
//  centers, and writes the results into `res.txt` with the same format.
//
//  * You may (and should) include some extra headers for optimizations.
//
//  * You should and ONLY should modify the function body of `kmeans()`.
//    DO NOT change any other existing part of the program.
//
//  * You may add your own auxiliary functions if you wish. Extra declarations
//    can go in `kmeans.h`.
//
// Jose @ ShanghaiTech University
//
////////////////

#include <fstream>
#include <limits>
#include <math.h>
#include <chrono>
#include "kmeans.h"


/*********************************************************
        Your extra headers and static declarations
 *********************************************************/
#include <assert.h>
//#include <omp.h>


/*********************************************************
                           End
 *********************************************************/


/*
 * Entrance point. Time ticking will be performed, so it will be better if
 *   you have cleared the cache for precise profiling.
 *
 */
int
main (int argc, char *argv[])
{
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input.txt> <output.txt>"
                  << std::endl;
        return -1;
    }
    if (!(bool)std::ifstream(argv[1])) {
        std::cerr << "ERROR: Data file " << argv[1] << " does not exist!"
                  << std::endl;
        return -1;
    }
    if ((bool)std::ifstream(argv[2])) {
        std::cerr << "ERROR: Destination " << argv[2] << " already exists!"
                  << std::endl;
        return -1;
    }
    FILE *fi = fopen(argv[1], "r"), *fo = fopen(argv[2], "w");

    /* From `ori.txt`, acquire dataset size, number of colors (i.e. K in
       K-Means),and read in all data points into static array `data`. */
    int pn, cn;

    assert(fscanf(fi, "%d / %d\n", &pn, &cn) == 2);

    // std::cout << "cn = " << cn << std::endl;

    point_t * const data = new point_t[pn];
    color_t * const coloring = new color_t[pn];

    for (int i = 0; i < pn; ++i)
        coloring[i] = 0;

    int i = 0, c;
    double x, y;

    while (fscanf(fi, "%lf, %lf, %d\n", &x, &y, &c) == 3) {
        data[i++].setXY(x, y);
        if (c < 0 || c >= cn) {
            std::cerr << "ERROR: Invalid color code encoutered!"
                      << std::endl;
            return -1;
        }
    }
    if (i != pn) {
        std::cerr << "ERROR: Number of data points inconsistent!"
                  << std::endl;
        return -1;
    }

    /* Generate a random set of initial center points. */
    point_t * const mean = new point_t[cn];

    srand(5201314);
    for (int i = 0; i < cn; ++i) {
        int idx = rand() % pn;
        mean[i].setXY(data[idx].getX(), data[idx].getY());
    }

    /* Invode K-Means algorithm on the original dataset. It should cluster
       the data points in `data` and assign their color codes to the
       corresponding entry in `coloring`, using `mean` to store the center
       points. */
    std::cout << "Doing K-Means clustering on " << pn
              << " points with K = " << cn << "..." << std::flush;
    auto ts = std::chrono::high_resolution_clock::now();
    kmeans(data, mean, coloring, pn, cn);
    auto te = std::chrono::high_resolution_clock::now();
    std::cout << "done." << std::endl;
    std::cout << " Total time elapsed: "
              << std::chrono::duration_cast<std::chrono::milliseconds> \
                 (te - ts).count()
              << " milliseconds." << std::endl;

    /* Write the final results to `res.txt`, in the same format as input. */
    fprintf(fo, "%d / %d\n", pn, cn);
    for (i = 0; i < pn; ++i)
        fprintf(fo, "%.8lf, %.8lf, %d\n", data[i].getX(), data[i].getY(),
                coloring[i]);

    /* Free the resources and return. */
    delete[](data);
    delete[](coloring);
    delete[](mean);
    fclose(fi);
    fclose(fo);
    return 0;
}


/*********************************************************
           Feel free to modify the things below
 *********************************************************/

/*
 * K-Means algorithm clustering. Originally implemented in a traditional
 *   sequential way. You should optimize and parallelize it for a better
 *   performance. Techniques you can use include but not limited to:
 *
 *     1. OpenMP shared-memory parallelization. (Used)
 *     2. SSE SIMD instructions. (...)
 *     3. Cache optimizations. (Maybe not)
 *     4. Manually using pthread. (No No No)
 *     5. remove sqrt (easy)
 *     6. Algorithm optimization?
 *     7. Space for time (use array)
 *     8. ...
 *
 */
void
kmeans (point_t * const data, point_t * const mean, color_t * const coloring,
        const int pn, const int cn)
{
    bool converge = true; // if the algorithm converges
    double* const sumx_array = new double[cn]; // Use an array to store the sum of all the x coordinate of each color
    double* const sumy_array = new double[cn]; // Use an array to store the sum of all the y coordinate of each color
    int* const count_array = new int[cn]; // Use an array to store the count

    /* Loop through the following two stages until no point changes its color
       during an iteration. */

    //int loop = 0;
    do {
        converge = true;
        // Initialize the 3 arrays
        for (int i = 0; i < cn; ++i)
            sumx_array[i] = 0;

        for (int i = 0; i < cn; ++i)
            sumy_array[i] = 0;

        for (int i = 0; i < cn; ++i)
            count_array[i] = 0;

        /* Compute the color of each point. A point gets assigned to the
           cluster with the nearest center point. */

        // #pragma omp parallel
        // {
        // #pragma omp for
        for (int i = 0; i < pn; ++i) { // loop over pn data points
            color_t new_color = cn; // number of colors
            double min_dist = std::numeric_limits<double>::infinity();

            for (color_t c = 0; c < cn; ++c) {
                double dist = pow(data[i].getX() - mean[c].getX(), 2) + pow(data[i].getY() - mean[c].getY(), 2);
                if (dist < min_dist) {
                    min_dist = dist;
                    new_color = c;
                }
            }

            // std::cout << ", new_color = " << new_color << std::endl;

            // #pragma omp critical

            sumx_array[new_color] += data[i].getX();
            sumy_array[new_color] += data[i].getY();
            count_array[new_color]++;


            if (coloring[i] != new_color) {
                // std::cout << "new_color=" << new_color << ", i = " << i << std::endl;
                /* std::cout << "data[i].getX()=" << data[i].getX() << std::endl;
                std::cout << "data[i].getY()=" << data[i].getY() << std::endl;
                std::cout << "set sumx_array[new_color]=" << sumx_array[new_color] << std::endl;
                std::cout << "set sumx_array[new_color]=" << sumx_array[new_color] << std::endl;
                std::cout << "set sumy_array[new_color]=" << sumy_array[new_color] << std::endl;
                std::cout << "set count_array[new_color]=" << count_array[new_color] << std::endl; */
                coloring[i] = new_color;
                converge = false;
            }
        }
        // }

        /* Calculate the new mean for each cluster to be the current average
           of point positions in the cluster. */

        /*
        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < pn; i++) {
                for (int c = 0; c < cn; c++) {
                    if (coloring[i] == c) {
                        #pragma omp critical
                        {
                            sumx_array[c] += data[i].getX();
                            sumy_array[c] += data[i].getY();
                            count_array[c]++;
                        }
                        break;
                    }
                }
            }
        }
        */

        // #pragma omp parallel for
        for (color_t c = 0; c < cn; ++c) {
            /* std::cout << "c=" << c << std::endl;
            std::cout << "mean[c]=" << mean[c] << std::endl;
            std::cout << "sumx_array[c]=" << sumx_array[c] << std::endl;
            std::cout << "sumy_array[c]=" << sumy_array[c] << std::endl;
            std::cout << "count_array[c]=" << count_array[c] << std::endl; */
            mean[c].setXY(sumx_array[c] / count_array[c], sumy_array[c] / count_array[c]);
        }

        // original code
        /*
        for (color_t c = 0; c < cn; ++c) {
            double sum_x = 0, sum_y = 0;
            int count = 0;

            #pragma omp parallel for reduction(+:sum_x, sum_y, count)
            for (int i = 0; i < pn; ++i) {
                if (coloring[i] == c) {
                    sum_x += data[i].getX();
                    sum_y += data[i].getY();
                    count++;
                }
            }

            mean[c].setXY(sum_x / count, sum_y / count);
        }
        */

    } while (!converge);

    delete sumx_array;
    delete sumy_array;
    delete count_array;
}

/*********************************************************
                           End
 *********************************************************/
