#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>
#include <dlib/gui_widgets.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <fstream>

using namespace dlib;
using namespace std;
//  ----------------------------------------------------------------------------
class gui : public drawable_window 
{
public:
  
    gui() : // All widgets take their parent window as an argument to their constructor.
      background(*this), mainView(*this), 
      photoView1(*this), photoView2(*this), photoView3(*this), 
      labelView1(*this), labelView2(*this), labelView3(*this)
    {
      loadIDPhotos();
      loadIDLabels();

      
      // background
      matrix<rgb_pixel> img_bg;
      load_image(img_bg, "../imgs/background.jpg");
      background.set_pos(0,0);
      background.set_image(img_bg);
      
      // main view: [10, 10, 810]
      load_image(img_init, "../imgs/init.jpg");
      mainView.set_pos(img_x, img_y);
      set_imgs(&mainView, img_w, img_h, img_init);
      
      // side view: [10, 10, 810]
      // size_t side_pos_x = 1124;size_t side_pos_y = 200;size_t side_size = 150;
      photoView1.set_pos(photo_x, photo_y);
      set_imgs(&photoView1, photo_w, photo_h, img_init);
      photoView2.set_pos(photo_x, photo_y + photo_h + photo_ym);
      set_imgs(&photoView2, photo_w, photo_h, img_init);
      photoView3.set_pos(photo_x, photo_y + 2*photo_h + 2*photo_ym);
      set_imgs(&photoView3, photo_w, photo_h, img_init);

      labelView1.set_pos(label_x, label_y);
      set_imgs(&labelView1, label_w, label_h, img_init);
      labelView2.set_pos(label_x, label_y + label_h + photo_ym);
      set_imgs(&labelView2, label_w, label_h, img_init);
      labelView3.set_pos(label_x, label_y + 2*label_h + 2*photo_ym);
      set_imgs(&labelView3, label_w, label_h, img_init);
      
      // set the size of this window
      set_size(1920,1080);
      set_title("Inspur Face Recognition System");
      show();
    } 


  ~gui() {
    close_window();
  }


  void updateGui( cv_image<bgr_pixel> & img_mV, 
		  std::vector<rectangle> rects,
		  std::vector<string> names
		  )
  {
    std::list<string> views;
    std::list<string>::iterator iter;
    for(size_t i = 0; i < names.size(); i++)
      views.push_back(names[i]);
    views.unique();

    float scale_w = 1.42857, scale_h = 1.42857;
    for(size_t i = 0; i < rects.size(); i++){
      draw_rectangle(img_mV,
		     //rects[i],
		     rectangle(rects[i].left()*scale_w, rects[i].top()*scale_h, rects[i].right()*scale_w, rects[i].bottom()*scale_h),
		     rgb_pixel(255,0,0));
    }
    mainView.set_image(img_mV);

    iter = views.begin();
    set_imgs(&photoView1, photo_w, photo_h, iter!= views.end()? IDPhotos[*iter++]:img_init);
    set_imgs(&photoView2, photo_w, photo_h, iter!= views.end()? IDPhotos[*iter++]:img_init);
    set_imgs(&photoView3, photo_w, photo_h, iter!= views.end()? IDPhotos[*iter++]:img_init);
    iter = views.begin();
    set_imgs(&labelView1, label_w, label_h, iter!= views.end()? IDLabels[*iter++]:img_init);
    set_imgs(&labelView2, label_w, label_h, iter!= views.end()? IDLabels[*iter++]:img_init);
    set_imgs(&labelView3, label_w, label_h, iter!= views.end()? IDLabels[*iter++]:img_init);
  }
private:
  image_widget  background;
  image_widget  mainView;
  image_widget  photoView1, photoView2, photoView3;
  image_widget  labelView1, labelView2, labelView3;


  matrix<rgb_pixel> img_init;
  // Dict for idenfication photos
  std::map<string, matrix<rgb_pixel>> IDPhotos;
  std::map<string, matrix<rgb_pixel>> IDLabels;

  size_t photo_x = 1430, photo_y = 250,
    photo_xm = 20, photo_ym = 60,
    photo_w = 160, photo_h = 200,
    img_w = 1280, img_h = 720,
    img_x = 60, img_y = 250,
    label_x = 1610, label_y = 250,
    label_w = 160, label_h = 200;
  

  void set_imgs(image_widget * img_widget,
		size_t width, size_t height,
		matrix<rgb_pixel> & img)
  {
    matrix<rgb_pixel> img_o;
    set_image_size(img_o, height, width);
    resize_image(img, img_o);
    img_widget->set_image(img_o);
  }

  void loadIDPhotos()
  {
    // load person images for display.
    cout << "Load identification photos..."<< endl;
    ifstream dBPhotos("../database/IDPhotos/IDPhotos.txt");
    string id_name, id_dir;
    std::vector<string> ID_names;
    std::vector<string> ID_dirs;
    while (dBPhotos >> id_name )
      {
	ID_names.push_back(id_name);
	dBPhotos >> id_dir;
	ID_dirs.push_back(id_dir);
      }
    
    matrix<rgb_pixel> id_img;
    for(size_t i = 0; i < ID_names.size(); i++){
      load_image(id_img, ID_dirs[i]);
      IDPhotos[ID_names[i]] = id_img;
    }
    cout<< ID_names.size() << " identification photos loaded."<<endl;
  }

  void loadIDLabels()
  {
    // load person images for display.
    cout << "Load identification labels..."<< endl;
    ifstream dBLabels("../database/IDLabels/IDLabels.txt");
    string id_name, id_dir;
    std::vector<string> ID_names;
    std::vector<string> ID_dirs;
    while (dBLabels >> id_name )
      {
	ID_names.push_back(id_name);
	dBLabels >> id_dir;
	ID_dirs.push_back(id_dir);
      }
    
    matrix<rgb_pixel> id_img;
    for(size_t i = 0; i < ID_names.size(); i++){
      load_image(id_img, ID_dirs[i]);
      IDLabels[ID_names[i]] = id_img;
    }
    cout<< ID_names.size() << " identification labels loaded."<<endl;
  }
};
