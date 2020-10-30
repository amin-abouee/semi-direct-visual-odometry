// feature_selection.cpp

// void FeatureSelection::computeGradient( const cv::Mat& currentTemplateImage,
//                                             cv::Mat& templateGradientX,
//                                             cv::Mat& templateGradientY )
// {
//     int h = currentTemplateImage.rows;
//     int w = currentTemplateImage.cols;

//     // ALIGNMENT_LOG( DEBUG ) << "h: " << h << ", w: " << w;
//     // [1, 1; w-1, h-1]
//     for ( int y( 1 ); y < h - 1; y++ )
//     {
//         for ( int x( 1 ); x < w - 1; x++ )
//         {
//             templateGradientX.at< float >( y, x ) =
//               0.5 * ( currentTemplateImage.at< float >( y, x + 1 ) - currentTemplateImage.at< float >( y, x - 1 ) );
//             templateGradientY.at< float >( y, x ) =
//               0.5 * ( currentTemplateImage.at< float >( y + 1, x ) - currentTemplateImage.at< float >( y - 1, x ) );
//         }
//     }

//     // ALIGNMENT_LOG( DEBUG ) << "center computed";

//     // for first and last rows
//     for ( int x( 1 ); x < w - 1; x++ )
//     {
//         templateGradientX.at< float >( 0, x ) =
//           0.5 * ( currentTemplateImage.at< float >( 0, x + 1 ) - currentTemplateImage.at< float >( 0, x - 1 ) );
//         templateGradientY.at< float >( 0, x ) =
//           0.5 * ( currentTemplateImage.at< float >( 1, x ) - currentTemplateImage.at< float >( 0, x ) );

//         templateGradientX.at< float >( h - 1, x ) =
//           0.5 * ( currentTemplateImage.at< float >( h - 1, x + 1 ) - currentTemplateImage.at< float >( h - 1, x - 1 ) );
//         templateGradientY.at< float >( h - 1, x ) =
//           0.5 * ( currentTemplateImage.at< float >( h - 1, x ) - currentTemplateImage.at< float >( h - 2, x ) );
//     }

//     // ALIGNMENT_LOG( DEBUG ) << "first and last rows";

//     // for first and last cols
//     for ( int y( 1 ); y < h - 1; y++ )
//     {
//         templateGradientX.at< float >( y, 0 ) =
//           0.5 * ( currentTemplateImage.at< float >( y, 1 ) - currentTemplateImage.at< float >( y, 0 ) );
//         templateGradientY.at< float >( y, 0 ) =
//           0.5 * ( currentTemplateImage.at< float >( y + 1, 0 ) - currentTemplateImage.at< float >( y - 1, 0 ) );

//         templateGradientX.at< float >( y, w - 1 ) =
//           0.5 * ( currentTemplateImage.at< float >( y, w - 1 ) - currentTemplateImage.at< float >( y, w - 2 ) );
//         templateGradientY.at< float >( y, w - 1 ) =
//           0.5 * ( currentTemplateImage.at< float >( y + 1, w - 1 ) - currentTemplateImage.at< float >( y - 1, w - 1 ) );
//     }

//     // ALIGNMENT_LOG( DEBUG ) << "first and last cols";

//     // upper left
//     templateGradientX.at< float >( 0, 0 ) =
//       0.5 * ( currentTemplateImage.at< float >( 0, 1 ) - currentTemplateImage.at< float >( 0, 0 ) );
//     // upper right
//     templateGradientX.at< float >( 0, w - 1 ) =
//       0.5 * ( currentTemplateImage.at< float >( 0, w - 1 ) - currentTemplateImage.at< float >( 0, w - 2 ) );
//     // lower left
//     templateGradientX.at< float >( h - 1, 0 ) =
//       0.5 * ( currentTemplateImage.at< float >( h - 1, 1 ) - currentTemplateImage.at< float >( h - 1, 0 ) );
//     // lower right
//     templateGradientX.at< float >( h - 1, w - 1 ) =
//       0.5 * ( currentTemplateImage.at< float >( h - 1, w - 1 ) - currentTemplateImage.at< float >( h - 1, w - 2 ) );

//     // upper left
//     templateGradientY.at< float >( 0, 0 ) =
//       0.5 * ( currentTemplateImage.at< float >( 1, 0 ) - currentTemplateImage.at< float >( 0, 0 ) );
//     // upper right
//     templateGradientY.at< float >( 0, w - 1 ) =
//       0.5 * ( currentTemplateImage.at< float >( 1, w - 1 ) - currentTemplateImage.at< float >( 0, w - 1 ) );
//     // lower left
//     templateGradientY.at< float >( h - 1, 0 ) =
//       0.5 * ( currentTemplateImage.at< float >( h - 1, 0 ) - currentTemplateImage.at< float >( h - 2, 0 ) );
//     // lower right
//     templateGradientY.at< float >( h - 1, w - 1 ) =
//       0.5 * ( currentTemplateImage.at< float >( h - 1, w - 1 ) - currentTemplateImage.at< float >( h - 2, w - 1 ) );
// }


 // struct gridData
// {
//     int x;
//     int y;
//     float max;
//     gridData()
//     {
//         x = -1;
//         y = -1;
//         max = 0.0;
//     }
// };

// std::vector <gridData> stack(cols);
// std::vector < std::vector <gridData> > table(rows, stack);

// float* pixelPtr = m_gradientMagnitude.ptr<float>();
// for(int i(0); i< m_gradientMagnitude.rows; i++)
// {
//     for(int j(0); j< m_gradientMagnitude.cols; j++, pixelPtr++)
//     {
//         const int indy = i / gridSize;
//         const int indx = j / gridSize;
//         if (*pixelPtr > table[indy][indx].max)
//         {
//             table[indy][indx].max = *pixelPtr;
//             table[indy][indx].x = j;
//             table[indy][indx].y = i;
//         }
//     }
// }

// for(int i(0); i<table.size(); i++)
// {
//     for (int j(0); j<table[i].size(); j++)
//     {
//         // std::cout << "row id: " << table[i][j].y << ", col id: " << table[i][j].x << ", max: " <<  table[i][j].max << std::endl;
//         const auto x = table[i][j].x;
//         const auto y = table[i][j].y;
//         const auto max = table[i][j].max;
//         if (max > 0)
//         {
//             std::unique_ptr< Feature > feature = std::make_unique< Feature >(
//                 frame, Eigen::Vector2d( x, y ),
//                 m_gradientMagnitude.at< float >( y, x ),
//                 m_gradientOrientation.at< float >( y, x ), 0 );
//             frame.addFeature(feature);
//         }
//     }
// }

// FeatureSelection::FeatureSelection( const cv::Mat& imgGray )
// {
//     // featureLogger = spdlog::stdout_color_mt( "FeatureSelection" );
//     // featureLogger->set_level( spdlog::level::debug );
//     // featureLogger->set_pattern( "[%Y-%m-%d %H:%M:%S] [%s:%#] [%n->%l] [thread:%t] %v" );

//     // https://answers.opencv.org/question/199237/most-accurate-visual-representation-of-gradient-magnitude/
//     // https://answers.opencv.org/question/136622/how-to-calculate-gradient-in-c-using-opencv/
//     // https://docs.opencv.org/master/d5/d0f/tutorial_py_gradients.html
//     // https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
//     // http://ninghang.blogspot.com/2012/11/list-of-mat-type-in-opencv.html

//     int ddepth     = CV_32F;
//     int ksize      = 1;
//     double scale   = 1.0;
//     double delta   = 0.0;
//     int borderType = cv::BORDER_DEFAULT;

//     // const cv::Mat imgGray = frame.m_imagePyramid.getBaseImage();
//     // auto t1 = std::chrono::high_resolution_clock::now();
//     // cv::Mat dx, absDx;
//     cv::Sobel( imgGray, m_dx, ddepth, 1, 0, ksize, scale, delta, borderType );
//     // cv::convertScaleAbs( dx, absDx );

//     // cv::Mat dy, absDy;
//     cv::Sobel( imgGray, m_dy, CV_32F, 0, 1, ksize, scale, delta, borderType );

//     // m_dx = cv::Mat ( imgGray.size(), CV_32F );
//     // m_dy = cv::Mat ( imgGray.size(), CV_32F );
//     // computeGradient(imgGray, m_dx, m_dy);

//     // cv::Mat mag, angle;
//     cv::cartToPolar( m_dx, m_dy, m_gradientMagnitude, m_gradientOrientation, true );
//     // auto t2 = std::chrono::high_resolution_clock::now();
//     // std::cout << "Elapsed time for gradient magnitude: " << std::chrono::duration_cast< std::chrono::milliseconds >( t2 - t1 ).count()
//     //   << std::endl;

//     // featureLogger->info("Elapsed time for gradient magnitude: {}", std::chrono::duration_cast< std::chrono::milliseconds >( t2 - t1
//     // ).count());

//     Feature_Log( DEBUG ) << "Init Feature Selection";
// }

// FeatureSelection::FeatureSelection( const cv::Mat& imgGray )
// {
//     m_imgGray = std::make_shared< cv::Mat >( imgGray );
//     m_features.reserve(2000);
// }

// FeatureSelection::FeatureSelection( const cv::Mat& imgGray, const uint32_t numberFeatures ):
// m_numberFeatures(numberFeatures)
// {
//     m_imgGray = std::make_shared< cv::Mat >( imgGray );
//     m_features.reserve(numberFeatures * 2);
// }