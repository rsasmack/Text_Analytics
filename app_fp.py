from flask import Flask, render_template, request, jsonify
from flask import render_template_string
from flask_cors import CORS
import pickle
import smtplib
from email.message import EmailMessage
from PIL import Image
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from bs4 import BeautifulSoup
import re
from torch import nn
from torch.nn import Dropout

# Initialize the tokenizer and the model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Add dropout layer in the model
model.classifier = nn.Sequential(
    Dropout(0.5),
    nn.Linear(model.config.dim, model.config.num_labels)
)

# Checking if CUDA is available and setting device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
model = model.to(device)

# Load the trained model weights
model.load_state_dict(torch.load('distilbert50_v3.pth', map_location=device))

def make_prediction(question1, question2):
    inputs = tokenizer.encode_plus(
        question1, question2, add_special_tokens=True, max_length=128, padding='max_length', truncation='only_second'
    )
    input_ids = torch.tensor([inputs['input_ids']], dtype=torch.long).to(device)
    attention_mask = torch.tensor([inputs['attention_mask']], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Get the predicted class index
    pred_class_idx = torch.argmax(outputs.logits).item()

    # Return a descriptive output
    if pred_class_idx == 0:
        return "The questions are not duplicates"
    else:
        return "The questions are duplicates"

app = Flask(__name__)

@app.route('/index.html')
def index():
    return render_template_string(index_html_code)

# index.html code as a string
index_html_code = '''
    <!DOCTYPE html>
    <html lang="en">

    <head>
      <meta charset="utf-8">
      <meta content="width=device-width, initial-scale=1.0" name="viewport">

      <title>NLP Project</title>
      <meta content="" name="description">
      <meta content="" name="keywords">

      <!-- Favicons -->
      <link href="static/static/assets/img/favicon.png" rel="icon">
      <link href="static/static/assets/img/apple-touch-icon.png" rel="apple-touch-icon">

      <!-- Fonts -->
      <link href="https://fonts.googleapis.com" rel="preconnect">
      <link href="https://fonts.gstatic.com" rel="preconnect" crossorigin>
      <link href="https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,300;1,400;1,500;1,600;1,700;1,800&family=Montserrat:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&family=Lato:ital,wght@0,100;0,300;0,400;0,700;0,900;1,100;1,300;1,400;1,700;1,900&display=swap" rel="stylesheet">

      <!-- Vendor CSS Files -->
      <link href="static/assets/vendor/bootstrap/css/bootstrap.min.css" rel="stylesheet">
      <link href="static/assets/vendor/bootstrap-icons/bootstrap-icons.css" rel="stylesheet">
      <link href="static/assets/vendor/glightbox/css/glightbox.min.css" rel="stylesheet">
      <link href="static/assets/vendor/swiper/swiper-bundle.min.css" rel="stylesheet">
      <link href="static/assets/vendor/aos/aos.css" rel="stylesheet">

      <!-- Template Main CSS File -->
      <link href="static/assets/css/main.css" rel="stylesheet">

    </head>

    <body class="index-page" data-bs-spy="scroll" data-bs-target="#navmenu">

      <!-- ======= Header ======= -->
      <header id="header" class="header fixed-top d-flex align-items-center">
        <div class="container-fluid d-flex align-items-center justify-content-between">

          <a href="index.html" class="logo d-flex align-items-center me-auto me-xl-0">
            <!-- Uncomment the line below if you also wish to use an image logo -->
            <!-- <img src="static/assets/img/logo.png" alt=""> -->
            <h1>Duplicase</h1>
            <span>.</span>
          </a>

          <!-- Nav Menu -->
          <nav id="navmenu" class="navmenu">
            <ul>
              <li><a href="/#hero" class="active">Home</a></li>
              <li><a href="/#features">About</a></li>
              <li><a href="/#product">Product</a></li>
              <li><a href="/#team">Team</a></li>
              <li><a href="/#contact">Contact</a></li>
            </ul>

            <i class="mobile-nav-toggle d-xl-none bi bi-list"></i>
          </nav><!-- End Nav Menu -->



        </div>
      </header><!-- End Header -->

      <main id="main">

        <!-- Hero Section - Home Page -->
        <section id="hero" class="hero">

          <img src="static/assets/img/stats-bg.jpg" alt="" data-aos="fade-in">

          <div class="container">
            <div class="row">
              <div class="col-lg-10">
                <h2 data-aos="fade-up" data-aos-delay="100">Welcome to Our Website</h2>
                <p data-aos="fade-up" data-aos-delay="200">We are team of talented AI Professionals.Working at Different Products across Machine Learning and Artificial Intelligence </p>
              </div>
              <div class="col-lg-5">
                <form action="#" class="sign-up-form d-flex" data-aos="fade-up" data-aos-delay="300">
                  <input type="text" class="form-control" placeholder="Enter email address">
                  <input type="submit" class="btn btn-primary" value="Sign up">
                </form>
              </div>
            </div>
          </div>

        </section><!-- End Hero Section -->



          <!-- About Section - Home Page -->
        <section id="features" class="features">

          <!--  Section Title -->
          <div class="container section-title" data-aos="fade-up">
            <h3>About Us</h3>
                <h2>Contents---Write Somethings</h2>
                <p>Misty Pebble Technology Services is one of India’s fastest-growing IT companies, recently starting its operation in Delhi. Misty Pebble provides Information Technology-related services to the global market, with a prime emphasis on Project Management, Software development and Maintenance, ERP Implementations, Web Development, and Technical Support/Services. We believe that success comes through hard work, dedication, and perseverance. We highly value the abilities, perspectives, and innovative qualities of people. We promote freethinking and offer extreme challenges so that people can excel and grow continuously.</p>

          </div><!-- End Section Title -->

          <div class="container">

            <div class="row gy-4 align-items-center features-item">
              <div class="col-lg-5 order-2 order-lg-1" data-aos="fade-up" data-aos-delay="200">
                <h3> Our Vision</h3>
                 <p>Our vision is to be a well-established software consulting & development
                    company to serve the Agencies and Enterprises. We are emerged as a globally
                    recognized software development company by providing the superior quality solutions.
                    As a committed team we shall strive for being integral part of our customer’s success
                    by providing value based Technical solutions. Promoting freethinking and offer extreme
                    challenges so that people can excel and grow continuously. Exploring new opportunities and
                    areas for the growth of our customers and our organization.</p>

                <!---<a href="#" class="btn btn-get-started">Get Started</a> -->
              </div>
              <div class="col-lg-7 order-1 order-lg-2 d-flex align-items-center" data-aos="zoom-out" data-aos-delay="100">
                <div class="image-stack">
                  <!--- <img src="static/assets/img/features-light-1.jpg" alt="" class="stack-front">-->
                  <img src="static/assets/img/features-light-1.jpg" alt="" class="stack-back">
                </div>
              </div>
            </div><!-- Features Item -->

            <div class="row gy-4 align-items-stretch justify-content-between features-item ">
              <div class="col-lg-6 d-flex align-items-center features-img-bg" data-aos="zoom-out">
                <img src="static/assets/img/features-light-2.jpg" class="img-fluid" alt="">
              </div>
              <div class="col-lg-5 d-flex justify-content-center flex-column" data-aos="fade-up">
                 <h3>Our Mission</h3>
                 <p>We are working with a mission to provide customer-centric, result-oriented,
                 cost-competitive IT & software solutions to our valuable global clients.Constant
                 innovation is our main key for achieving the ultimate goal of success. Misty pebble
                 wants to be a dependable world-class organization by delivering Superior Quality Software
                 Services, Solutions and Products by leveraging, People, Processes and Technologies.
                 We shall achieve this Quality Service by comprehending their need through close interaction and by creating a global network.</p>

              </div>
            </div><!-- Features Item -->

            <div class="row gy-4 align-items-center features-item">
              <div class="col-lg-5 order-2 order-lg-1" data-aos="fade-up" data-aos-delay="200">
                <h3>Core Values</h3>
                 <p>We are working with a mission to provide customer-centric, result-oriented,
                 cost-competitive IT & software solutions to our valuable global clients. Constant
                 innovation is our main key for achieving the ultimate goal of success. Misty pebble
                 wants to be a dependable world-class organization by delivering Superior Quality Software
                 Services, Solutions and Products by leveraging, People, Processes and Technologies.
                 We shall achieve this Quality Service by comprehending their need through close interaction and by creating a global network.</p>


                <!---<a href="#" class="btn btn-get-started">Get Started</a> -->
              </div>
              <div class="col-lg-7 order-1 order-lg-2 d-flex align-items-center" data-aos="zoom-out" data-aos-delay="100">
                <div class="image-stack">
                  <!--- <img src="static/assets/img/features-light-1.jpg" alt="" class="stack-front">-->
                  <img src="static/assets/img/features-light-3.jpg" alt="" class="stack-back">
                </div>
              </div>
            </div><!-- Features Item -->



          </div>

        </section><!-- End About Section -->

    <!-- Recent-posts Section - Home Page -->
        <section id="product" class="product">

          <!--  Section Title -->
          <div class="container section-title" data-aos="fade-up">
            <h2>Our Products</h2>
            <p>Check out our amazing product offerings! We take pride in delivering high-quality and innovative text analytics products to our customers.</p>
          </div><!-- End Section Title -->

          <div class="container">

            <div class="row gy-4">

              <div class="col-xl-4 col-md-6" data-aos="fade-up" data-aos-delay="100">
                <article>

                  <div class="post-img">
                    <img src="static/assets/img/blog/blog-1.jpg" alt="" class="img-fluid">
                  </div>

                   <h2 class="title">Duplicate Questions</h2>
                   <a class="pta-btn" href="/#dupli-check">CHECK IT OUT</a>
                </article>
              </div><!-- End post list item -->

              <div class="col-xl-4 col-md-6" data-aos="fade-up" data-aos-delay="200">
                <article>

                  <div class="post-img">
                    <img src="static/assets/img/blog/blog-2.jpg" alt="" class="img-fluid">
                  </div>

                  <p class="post-category">Question Answer System</p>

                  <h2 class="title">
                    <a href=#>Coming Sooonnn..!!!</a>
                  </h2>


                </article>
              </div><!-- End post list item -->

              <div class="col-xl-4 col-md-6" data-aos="fade-up" data-aos-delay="300">
                <article>

                  <div class="post-img">
                    <img src="static/assets/img/blog/blog-3.jpg" alt="" class="img-fluid">
                  </div>

                  <p class="post-category">Speech Recognition </p>

                  <h2 class="title">
                    <a href=#>Coming Sooonnn..!!!</a>
                  </h2>


                </article>
              </div><!-- End post list item -->

              <div class="row gy-4">

              <div class="col-xl-4 col-md-6" data-aos="fade-up" data-aos-delay="100">
                <article>

                  <div class="post-img">
                    <img src="static/assets/img/blog/blog-4.jpg" alt="" class="img-fluid">
                  </div>

                  <p class="post-category">Image Generation</p>

                  <h2 class="title">
                    <a href="blog-details.html">Click Here To Check Out This Feature</a>
                  </h2>


                </article>
              </div><!-- End post list item -->

              <div class="col-xl-4 col-md-6" data-aos="fade-up" data-aos-delay="200">
                <article>

                  <div class="post-img">
                    <img src="static/assets/img/blog/blog-5.jpg" alt="" class="img-fluid">
                  </div>

                  <p class="post-category">Text Summarization</p>

                  <h2 class="title">
                    <a href=#>Coming Sooonnn..!!!</a>
                  </h2>


                </article>
              </div><!-- End post list item -->

              <div class="col-xl-4 col-md-6" data-aos="fade-up" data-aos-delay="300">
                <article>

                  <div class="post-img">
                    <img src="static/assets/img/blog/blog-6.jpg" alt="" class="img-fluid">
                  </div>

                  <p class="post-category">Sentiment Analysis  </p>

                  <h2 class="title">
                    <a href=#>Coming Sooonnn..!!!</a>
                  </h2>


                </article>
              </div><!-- End post list item -->

            </div><!-- End recent posts list -->

          </div>

        </section><!-- End Recent-posts Section -->



        <!-- Team Section - Home Page -->
        <section id="team" class="team">

          <!--  Section Title -->
          <div class="container section-title" data-aos="fade-up">
            <h2>Team</h2>
            <p>Meet the Developer Behind the amazing Products</p>
          </div><!-- End Section Title -->

          <div class="container">

            <div class="row gy-5">

              <div class="col-lg-4 col-md-6 member" data-aos="fade-up" data-aos-delay="100">
                <div class="member-img">
                  <img src="static/assets/img/team/team-1.jpg" class="img-fluid" alt="">
                  <div class="social">
                    <a href="https://twitter.com/agg_2008"><i class="bi bi-twitter"></i></a>
                    <a href="https://www.linkedin.com/in/ankitagg2008/"><i class="bi bi-linkedin"></i></a>
                  </div>
                </div>
                <div class="member-info text-center">
                  <h4>Ankit Kumar Aggarwal </h4>
                  <span>AI Graduate Student</span>
                  <p>Platform Independent Visualization Expert with almost 14+ Years of Extensive Experience in End to End Implementation and management of Business Intelligence, Data Warehousing and CEM analytics projects. Having Worked with No. Of Clients from Europe, US, Middle East & Asia Region. I’m Have Good Knowledge on Banking, Telecom, and Education & Call Centre Operations Domain. My Current Role Is to Provide Technical Solution, Architectural Solution, Administrating and Support & Development for Clients Working On Cognos, SSRS, SSAS, Microstrategy and Various Other BI Tools and Technology. </p>
                </div>
              </div><!-- End Team Member -->

              <div class="col-lg-4 col-md-6 member" data-aos="fade-up" data-aos-delay="300">
                <div class="member-img">
                  <img src="static/assets/img/team/team-2.jpg" class="img-fluid" alt="">
                  <div class="social">
                    <a href="https://twitter.com/Maurya_Kanchan2"><i class="bi bi-twitter"></i></a>
                    <a href="https://www.linkedin.com/in/kanchan-maurya-44926b107/"><i class="bi bi-linkedin"></i></a>
                  </div>
                </div>
                <div class="member-info text-center">
                  <h4>Kanchan Maurya</h4>
                  <span>AI Graduate Student</span>
                  <p>Currently pursuing Masters in AI at Yeshiva University, where my fascination for artificial intelligence has led me to explore the limitless possibilities it offers. I actively engage in AI-focused events and workshops to stay ahead of the latest advancements in this exciting field. As an intern at S&P Global, I am gaining hands-on experience in AI projects, applying my theoretical knowledge to real-world scenarios. I am passionate about using AI for social good and actively seek opportunities to make a positive impact on society through my work.</p>
                </div>
              </div><!-- End Team Member -->

              <div class="col-lg-4 col-md-6 member" data-aos="fade-up" data-aos-delay="300">
                <div class="member-img">
                  <img src="static/assets/img/team/team-3.jpg" class="img-fluid" alt="">
                  <div class="social">

                    <a href="https://www.linkedin.com/in/avinashrs/"><i class="bi bi-linkedin"></i></a>
                  </div>
                </div>
                <div class="member-info text-center">
                  <h4>Avinash RS </h4>
                  <span>AI Graduate Student</span>
                  <p>As a graduate student at Yeshiva University, I am pursuing my passion for data science and machine learning through a Master's degree in Artificial Intelligence. I am also an intern at S&P Global, where I apply my skills and knowledge in computer vision to develop innovative AI solutions As an AI enthusiast with a strong background in analytics and with acquired skillsets over the years, I add value to any organization focused on developing innovative AI/ML solutions using cutting-edge technologies.</p>
                </div>
              </div><!-- End Team Member -->

              <div class="col-lg-4 col-md-6 member" data-aos="fade-up" data-aos-delay="400">
                <div class="member-img">
                  <img src="static/assets/img/team/team-5.jpg" class="img-fluid" alt="">
                  <div class="social">
                    <a href="#"><i class="bi bi-twitter"></i></a>
                    <a href="#"><i class="bi bi-facebook"></i></a>
                    <a href="#"><i class="bi bi-instagram"></i></a>
                    <a href="#"><i class="bi bi-linkedin"></i></a>
                  </div>
                </div>
                <div class="member-info text-center">
                  <h4>Shengjie Zhao</h4>
                  <span>AI Graduate Student</span>
                  <p></p>
                </div>
              </div><!-- End Team Member -->


            </div>

          </div>

        </section><!-- End Team Section -->

        <!-- dupli-check Section - Home Page -->
        <section id="dupli-check" class="features">

        <div class="container">
            <div class="row justify-content-center" data-aos="zoom-in" data-aos-delay="100">
                <div class="col-xl-10">
                    <div class="text-center">
                        <form id="dupli-check-form" class="php-email-form" data-aos="fade-up" data-aos-delay="200">
    
                            <div>
                                <textarea id="q1" name="q1" class="form-control" placeholder="Question 1" required></textarea>
                            </div>
    
                            <br>
                            <br>
                            <div>
                                <textarea id="q2" name="q2" class="form-control" placeholder="Question 2" required></textarea>
                            </div>
    
                            <br>
                            <br>
                            <div class="btn-find" >
                                <button class="btn btn-primary" type="submit">Find</button>
                            </div>
    
                        </form>
                        <!-- Display the result here -->
                        <div class="result"></div>
                    </div>
                </div>
            </div>
        </div>
    
        </section><!-- End dupli-check Section -->
    
        <!-- Include jQuery -->
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    
        <!-- AJAX script -->
        <script>
        $(document).ready(function(){
            $("#dupli-check-form").on("submit", function(event){
                event.preventDefault();
                $.ajax({
                    url: '/find_duplicate_questions',
                    type: 'POST',
                    data: $(this).serialize(),
                    success: function(response){
                        // Update the page with the result
                        $(".result").html("<h3>Result:</h3><p>" + response.result + "</p>");
                    }
                });
            });
        });
        </script>



        <!-- Contact Section - Home Page -->
        <section id="contact" class="contact">

          <!--  Section Title -->
          <div class="container section-title" data-aos="fade-up">
            <h2>Contact</h2>
            <p>Hit us with any query troubling you..!!!</p>
          </div><!-- End Section Title -->

          <div class="container" data-aos="fade-up" data-aos-delay="100">

            <div class="row gy-4">

              <div class="col-lg-6">

                <div class="row gy-4">
                  <div class="col-md-6">
                    <div class="info-item" data-aos="fade" data-aos-delay="200">
                      <i class="bi bi-geo-alt"></i>
                      <h3>Address</h3>
                      <p>Yeshiva university</p>
                      <p>New York, NY 10033</p>
                    </div>
                  </div><!-- End Info Item -->

                  <div class="col-md-6">
                    <div class="info-item" data-aos="fade" data-aos-delay="300">
                      <i class="bi bi-telephone"></i>
                      <h3>Call Us</h3>
                      <p>+91 96436 08100</p>
                      <p>+1 201 630 1508</p>
                    </div>
                  </div><!-- End Info Item -->

                  <div class="col-md-6">
                    <div class="info-item" data-aos="fade" data-aos-delay="400">
                      <i class="bi bi-envelope"></i>
                      <h3>Email Us</h3>
                      <p>ankit.agg.2008@gmail.com</p>
                      <p>kanchanmaurya95@gmail.com</p>
                      <p>aeroapj@gmail.com</p>
                    </div>
                  </div><!-- End Info Item -->

                  <div class="col-md-6">
                    <div class="info-item" data-aos="fade" data-aos-delay="500">
                      <i class="bi bi-clock"></i>
                      <h3>Open Hours</h3>
                      <p>Monday - Friday</p>
                      <p>9:00AM - 05:00PM</p>
                    </div>
                  </div><!-- End Info Item -->

                </div>

              </div>

              <div class="col-lg-6">
                <form action="forms/contact.php" method="post" class="php-email-form" data-aos="fade-up" data-aos-delay="200">
                  <div class="row gy-4">

                    <div class="col-md-6">
                      <input type="text" name="name" class="form-control" placeholder="Your Name" required>
                    </div>

                    <div class="col-md-6 ">
                      <input type="email" class="form-control" name="email" placeholder="Your Email" required>
                    </div>

                    <div class="col-md-12">
                      <input type="text" class="form-control" name="subject" placeholder="Subject" required>
                    </div>

                    <div class="col-md-12">
                      <textarea class="form-control" name="message" rows="6" placeholder="Message" required></textarea>
                    </div>

                    <div class="col-md-12 text-center">
                      <div class="loading">Loading</div>
                      <div class="error-message"></div>
                      <div class="sent-message">Your message has been sent. Thank you!</div>

                      <button type="submit">Send Message</button>
                    </div>

                  </div>
                </form>
              </div><!-- End Contact Form -->

            </div>

          </div>

        </section><!-- End Contact Section -->

      </main>

      <!-- ======= Footer ======= -->
      <footer id="footer" class="footer">

        <div class="container footer-top">
          <div class="row gy-4">
            <div class="col-lg-5 col-md-12 footer-about">
              <a href="index.html" class="logo d-flex align-items-center">
                <span>Duplicase</span>
              </a>
              <p>NLP Project</p>
              <div class="social-links d-flex mt-4">
                <a href=""><i class="bi bi-twitter"></i></a>
                <a href=""><i class="bi bi-facebook"></i></a>
                <a href=""><i class="bi bi-instagram"></i></a>
                <a href=""><i class="bi bi-linkedin"></i></a>
              </div>
            </div>

            <div class="col-lg-2 col-6 footer-links">
              <h4>Useful Links</h4>
              <ul>
              <li><a href="index.html#hero">Home</a></li>
              <li><a href="index.html#features">About</a></li>
              <li><a href="index.html#product">Product</a></li>
              <li><a href="index.html#team">Team</a></li>
              <li><a href="index.html#contact">Contact</a></li>
              </ul>
            </div>

            <div class="col-lg-2 col-6 footer-links">
              <h4>Our Services</h4>
              <ul>
                <li><a href="index.html#product">Duplicate Questions</a></li>
                <li><a href="index.html#product">Question Answer</a></li>
                <li><a href="index.html#product">Speech Recognition</a></li>
                <li><a href="index.html#product">Image Generation</a></li>
                <li><a href="index.html#product">Sentiment Analysis</a></li>
              </ul>
            </div>

            <div class="col-lg-3 col-md-12 footer-contact text-center text-md-start">
              <h4>Contact Us</h4>
              <p>391 Central Avenue</p>
              <p>New Jersey, NJ 07307</p>
              <p>United States</p>
              <p class="mt-4"><strong>Phone:</strong> <span>+1 201-630-1508</span></p>
              <p><strong>Email:</strong> <span>kanchanmaurya95@gmail.com</span></p>
            </div>

          </div>
        </div>

        <div class="container copyright text-center mt-4">
          <p>&copy; <span>Copyright</span> <strong class="px-1">Duplicase</strong> <span>All Rights Reserved</span></p>
          <div class="credits">
            Designed by <a href="index.html#team">Duplicase</a>
          </div>
        </div>

      </footer><!-- End Footer -->

      <!-- Scroll Top Button -->
      <a href="#" id="scroll-top" class="scroll-top d-flex align-items-center justify-content-center"><i class="bi bi-arrow-up-short"></i></a>

      <!-- Preloader -->
      <div id="preloader">
        <div></div>
        <div></div>
        <div></div>
        <div></div>
      </div>

      <!-- Vendor JS Files -->
      <script src="static/assets/vendor/bootstrap/js/bootstrap.bundle.min.js"></script>
      <script src="static/assets/vendor/glightbox/js/glightbox.min.js"></script>
      <script src="static/assets/vendor/purecounter/purecounter_vanilla.js"></script>
      <script src="static/assets/vendor/isotope-layout/isotope.pkgd.min.js"></script>
      <script src="static/assets/vendor/swiper/swiper-bundle.min.js"></script>
      <script src="static/assets/vendor/aos/aos.js"></script>
      <script src="static/assets/vendor/php-email-form/validate.js"></script>

      <!-- Template Main JS File -->
      <script src="static/assets/js/main.js"></script>

    </body>

    </html>
'''


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/find_duplicate_questions', methods=['POST'])
def find_duplicate_questions():
    # Get user inputs from the form
    q1 = request.form['q1']
    q2 = request.form['q2']

    # Make prediction
    result = make_prediction(q1, q2)
    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True)





# Route to handle the form submission and find duplicate questions
# @app.route('/')
# def home():
#     return render_template_string(index_html_code)
#
# # Route to serve the index.html page
# @app.route('/find_duplicate_questions', methods=['POST'])
# def find_duplicate_questions():
#     # Get user inputs from the form
#     q1 = request.form['q1']
#     q2 = request.form['q2']
#
#     # Make prediction
#     result = make_prediction(q1, q2)
#     return render_template_string(index_html_code, result=result)


# # Route to handle the form submission and find duplicate questions
# @app.route('/find_duplicate_questions', methods=['POST'])
# def find_duplicate_questions():
#     # Get user inputs from the form
#     q1 = request.form['q1']
#     q2 = request.form['q2']
#
#     # Your code for generating the query point, making the prediction, and obtaining the result
#     query = helper.query_point_creator(q1, q2)  # Generate query point
#     result = model.predict(query)[0]  # Make prediction
#
#     return render_template_string(index_html_code, result=result)
#