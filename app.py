# Import the necessary modules and packages.
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

# Check if CUDA is available. CUDA is a parallel computing platform by Nvidia.
# It allows leveraging the GPU for computation, which is crucial for efficient deep learning.
# If CUDA is not available, fall back to CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
model = model.to(device)

# Load the trained model weights
model.load_state_dict(torch.load('distilbert50_v3.pth', map_location=device))

# Define a function to make a prediction using the model.
def make_prediction(question1, question2):
    # Prepare the inputs for the model. The inputs are tokenized,
    # which means they are transformed into a format the model can understand.
    # The input questions are limited to 128 tokens, with padding added if necessary.
    inputs = tokenizer.encode_plus(
        question1, question2, add_special_tokens=True, max_length=128, padding='max_length', truncation='only_second'
    )
    # Convert the inputs to PyTorch tensors and move them to the chosen device.
    input_ids = torch.tensor([inputs['input_ids']], dtype=torch.long).to(device)
    attention_mask = torch.tensor([inputs['attention_mask']], dtype=torch.long).to(device)

    # Switch the model to evaluation mode. This is necessary because some types of layers,
    # like dropout and batch normalization, behave differently during training and during evaluation.
    model.eval()

    # Make a forward pass through the model, i.e., make a prediction.
    # The 'with torch.no_grad()' context manager is used because gradients are not needed
    # during evaluation and this saves memory.
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Extract the predicted class index from the outputs.
    pred_class_idx = torch.argmax(outputs.logits).item()

    # Depending on the class index, return a descriptive output.
    if pred_class_idx == 0:
        return "The questions are not duplicates"
    else:
        return "The questions are duplicates"

app = Flask(__name__)

# Define a route for the URL '/index.html'. When this URL is accessed via a GET request,
# the 'index' function will be called and its return value will be the HTTP response.
@app.route('/index.html')
def index():
    # Render a HTML template from a string and return it.
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
      <link href="static/assets/img/favicon.png" rel="icon">
      <link href="static/assets/img/apple-touch-icon.png" rel="apple-touch-icon">
    
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
    <script>
      function showThanksMessage() { 
        // Get the form element
        var signUpForm = document.querySelector('.sign-up-form');
    
        // Hide the form
        signUpForm.style.display = 'none';
    
        // Create a new element for the thanks message
        var thanksMessage = document.createElement('p');
        thanksMessage.innerHTML = 'Thanks for signing up..!!!';
    
        // Append the message to the parent container
        var parentContainer = signUpForm.parentElement;
        parentContainer.appendChild(thanksMessage);
      }
      
      
    
    function displayTYContact() {
        // Get the form element
        var contactForm = document.querySelector('.php-email-form');
    
        // Hide the form
        contactForm.style.display = 'none';
    
        // Show the "Thank you" message
        var thankYouMessage = document.querySelector('.sent-message');
        thankYouMessage.style.display = 'block';
    }
    </script>
    
    </script>
    </head>
    
    <body class="index-page" data-bs-spy="scroll" data-bs-target="#navmenu">
    
      <!-- ======= Header ======= -->
      <header id="header" class="header fixed-top d-flex align-items-center">
        <div class="container-fluid d-flex align-items-center justify-content-between">
    
          <a href="index.html" class="logo d-flex align-items-center me-auto me-xl-0">
            <!-- Uncomment the line below if you also wish to use an image logo -->
            <!-- <img src="static/assets/img/logo.png" alt=""> -->
            <h1>MISTY PEBBLE</h1>
            <span></span>
          </a>
    
          <!-- Nav Menu -->
          <nav id="navmenu" class="navmenu">
            <ul>
              <li><a href="#hero" class="active">Home</a></li>
              <li><a href="#features">About</a></li>
              <li><a href="#product">Product</a></li>
              <li><a href="#team">Team</a></li>
              <li><a href="#contact">Contact</a></li>
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
                <h2 data-aos="fade-up" data-aos-delay="100">Welcome to the world of AI<h2>
                <p data-aos="fade-up" data-aos-delay="200">We are team of talented AI Professionals.Working at Different Products across Machine Learning and Artificial Intelligence </p>
              </div>
              <div class="col-lg-5">
                <form action="#" class="sign-up-form d-flex" data-aos="fade-up" data-aos-delay="300">
                  <input type="text" class="form-control" placeholder="Enter email address">
                  <input type="submit" class="btn btn-primary" value="Sign up" onclick="showThanksMessage()">
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
                 <!-- <h2>Contents---Write Somethings</h2> -->
                <p>Welcome to MISTY PEBBLE, where the future comes alive through Artificial Intelligence. We are an emerging AI product company, driven by an unwavering passion for innovation and a commitment to shaping tomorrow's technology landscape.
    
                    At MISTY PEBBLE, we're not just building AI products. We're crafting experiences that redefine industries and empower businesses. Our dynamic team of AI enthusiasts thrives on challenges and is dedicated to pushing the boundaries of what's possible.
    
                    Join us on this exciting journey as we navigate the uncharted territories of AI and pave the way for a smarter, more connected world. Together, let's turn AI into reality.</p>
    
          </div><!-- End Section Title -->
    
          <div class="container">
    
            <div class="row gy-4 align-items-center features-item">
              <div class="col-lg-5 order-2 order-lg-1" data-aos="fade-up" data-aos-delay="200">
                <h3> Our Vision</h3>
                 <p>Misty Pebble's vision is anchored in a future where human potential and AI innovation converge seamlessly. 
                 We aspire to be trailblazers in this transformative journey, envisioning a world where AI becomes an integral 
                 and ethical partner, enriching every facet of life. Our mission is to lead this evolution by creating AI solutions
                 that are not only cutting-edge but also prioritize empathy, responsibility, and sustainable progress.Beyond technology, 
                 Misty Pebble aims to foster a global community of AI enthusiasts, thinkers, and collaborators.
                 We seek to ignite a wave of positive change, propelling industries forward, addressing pressing global challenges,
                 and redefining the possibilities of human-AI coexistence. Together, we are shaping a future where AI transcends its 
                 role as a tool and becomes an empowering force that elevates societies and leaves an indelible mark on generations to come.
                </p>
    
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
                 <p>At Misty Pebble, we're on a mission to redefine possibilities through cutting-edge Artificial Intelligence. 
                 Our purpose is to craft innovative AI solutions that reshape industries, amplify human potential, and inspire
                 a future where technology knows no bounds.Driven by unwavering dedication, we're committed to setting new benchmarks
                 in AI excellence. By seamlessly integrating human ingenuity and AI prowess, we strive to empower individuals, businesses,
                 and society at large, driving progress and transformation.Our journey is defined by a commitment to unwavering ethics,
                 fostering a culture of collaboration, diversity, and relentless innovation. At Misty Pebble, we envision a horizon where
                 technology and humanity coalesce to create a world of limitless opportunities, one transformative pebble at a time</p>
               
              </div>
            </div><!-- Features Item -->
            
            <div class="row gy-4 align-items-center features-item">
              <div class="col-lg-5 order-2 order-lg-1" data-aos="fade-up" data-aos-delay="200">
                <h3>Core Values</h3>
                 <p>AAt Misty Pebble, our core values drive our every endeavor. We place paramount importance on being customer-centric, 
                 tailoring our AI solutions to meet and exceed the unique needs of our clients. This commitment fuels our unwavering dedication 
                 to delivering exceptional experiences and forging lasting relationships through active listening and continuous adaptation.
                 Innovation is the beating heart of Misty Pebble. We embrace challenges with a forward-looking perspective, channeling 
                 our energy into pushing the boundaries of AI possibilities. Our innovation-oriented ethos cultivates an environment where 
                 creativity thrives, enabling us to craft groundbreaking solutions that redefine industries, catalyze progress, and contribute 
                 to a future where technology knows no limits. Moreover, our dedication to being cost-effective underscores our belief that 
                 transformative AI solutions should be accessible without compromising quality. Through streamlined processes and strategic 
                 efficiency, we ensure that Misty Pebble remains a beacon of affordability, delivering value that empowers our partners and 
                 stakeholders to embrace the advantages of AI with confidence and optimism.</p>
               
    
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
    
                   <h2 class="title">DupliQuest</h2>
                   <a class="pta-btn" href="index.html#dupli-check">CHECK IT OUT</a>
                </article>
              </div><!-- End post list item -->
              
              
              
              <div class="col-xl-4 col-md-6" data-aos="fade-up" data-aos-delay="300">
                <article>
    
                  <div class="post-img">
                    <img src="static/assets/img/blog/blog-6.jpg" alt="" class="img-fluid">
                  </div>
    
                  <p class="post-category">Sentiment Analysis  </p>
    
                  <h2 class="title">
                  <a class="pta-btn" href="https://nlp-teachback.streamlit.app/" target="_blank">CHECK IT OUT</a>
                  </h2>
    
    
                </article>
              </div>
    
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
              </div>
    
              <!-- End post list item -->
              
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
                    <img src="static/assets/img/blog/blog-3.jpg" alt="" class="img-fluid">
                  </div>
    
                  <p class="post-category">Speech Recognition </p>
    
                  <h2 class="title">
                    <a href=#>Coming Sooonnn..!!!</a>
                  </h2>
    
    
                </article>
               </div>	
              <!-- End post list item -->
    
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
                  <p>AI Graduate Student | Entrepreneur | Ex-IBM | Infosys | Infovista| Hammer | Nagarro | AgreeYa | Nucleus | Datawarehouse And Business Intelligence Corporate Trainer</p>
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
                  <p>Intern @ S&P Global || Machine Learning Engineer || AI Enthusiast || Data Science || Computer Vision || Natural Language Processing || Ex-CGI  || Ex-Infosys </p>
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
                  <p>Intern @ S&P Global || Machine Learning Engineer || AI Enthusiast || Data Science || Computer Vision || Natural Language Processing</p>
                </div>
              </div><!-- End Team Member -->
    
              <div class="col-lg-4 col-md-6 member" data-aos="fade-up" data-aos-delay="400">
                <div class="member-img">
                  <img src="static/assets/img/team/team-5.jpg" class="img-fluid" alt="">
                
                </div>
                <div class="member-info text-center">
                  <h4>Shengjie Zhao</h4>
                  <span>AI Graduate Student</span>
                  <p> Machine Learning Engineer || AI Enthusiast || Data Science ||</p>
                </div>
              </div><!-- End Team Member -->
    
              
            </div>
    
          </div>
    
        </section><!-- End Team Section -->
    
        <!-- dupli-check Section - Home Page -->
            <!-- dupli-check Section - Home Page -->
            <section id="dupli-check" class="features">
        
        
            <div class="container" style="background-image: url('static/assets/img/cta-bg.jpg');">
                
        
                <div class="container section-title" data-aos="fade-up">
                <br>
                <h2>DupliQuest</h2>
                <br>
                <h4>Find whether both the questions are the same ???</h4>
                <br>
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
                        <p>Shahdara</p>
                        <p>Delhi, Pin code - 110093</p>
                        <p>India</p>
                    </div>
                  </div><!-- End Info Item -->
    
                  <div class="col-md-6">
                    <div class="info-item" data-aos="fade" data-aos-delay="300">
                      <i class="bi bi-telephone"></i>
                      <h3>Call Us</h3>
                      <p>+1 551 247 9855</p>
                      <p>+1 201 630 1508</p>
                      <p>+1 516 734 1732</p>
                      <p>+1 347 607 7502</p>
                    </div>
                  </div><!-- End Info Item -->
    
                  <div class="col-md-6">
                    <div class="info-item" data-aos="fade" data-aos-delay="400">
                      <i class="bi bi-envelope"></i>
                      <h3>Email Us</h3>
                      <p>aaggarwa@mail.yu.edu</p>
                      <p>kmaurya@mail.yu.edu</p>
                      <p>aswamina@mail.yu.edu</p>
                      <p>szhao3@mail.yu.edu</p>
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
                        <div class="col-md-6">
                          <input type="email" class="form-control" name="email" placeholder="Your Email" required>
                        </div>
                        <div class="col-md-12">
                          <input type="text" class="form-control" name="subject" placeholder="Subject" required>
                        </div>
                        <div class="col-md-12">
                          <textarea class="form-control" name="message" rows="6" placeholder="Message" required></textarea>
                        </div>
                        <div class="col-md-12 text-center">
                        
                          <div class="sent-message" style="display: none;">
                            Thank you! for your interest. Our team will reach out to you soon!
                          </div>
                          <button type="submit" onclick="displayTYContact()">Send Message</button>
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
                <span>MISTY PEBBLE</span>
              </a>
              
             
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
                <li><a href="index.html#product">DupliQuest</a></li>
                <li><a href="index.html#product">Sentiment Analysis</a></li>
                <li><a href="index.html#product">Question Answer</a></li>
                <li><a href="index.html#product">Speech Recognition</a></li>
                <li><a href="index.html#product">Image Generation</a></li>
                
              </ul>
            </div>
    
            <div class="col-lg-3 col-md-12 footer-contact text-center text-md-start">
              <h4>Contact Us</h4>
              <p>Shahdara</p>
              <p>Delhi, Pin code - 110093</p>
              <p>India</p>
            </div>
    
          </div>
        </div>
    
        <div class="container copyright text-center mt-4">
          <p>&copy; <span>Copyright</span> <strong class="px-1">MISTY PEBBLE</strong> <span>All Rights Reserved</span></p>
          <div class="credits">
            Designed by <a href="index.html#team">MISTY PEBBLE</a>
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

# Define a route for the root URL ("/"). When this URL is accessed via a GET request,
# the 'home' function will be called and its return value will be the HTTP response.
# This function renders the 'index.html' template and returns it as a response.
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the URL "/find_duplicate_questions". This route is accessed via a POST request,
# usually when a form is submitted on the client side.
@app.route('/find_duplicate_questions', methods=['POST'])
def find_duplicate_questions():
    # Get user inputs from the form using request.form.
    # These are the two questions that the user wants to compare.
    q1 = request.form['q1']
    q2 = request.form['q2']

    # Use the earlier defined make_prediction function to determine whether the questions are duplicates.
    result = make_prediction(q1, q2)

    # Return the result as a JSON response. This can be easily processed on the client side.
    # The 'jsonify' function in Flask returns a Flask.Response() object that has the
    # application/json mimetype.
    return jsonify({'result': result})


if __name__ == '__main__':
    # Start the Flask development server.
    # The debug parameter is set to False, meaning that changes to the code will not automatically
    # restart the server and exceptions will not be presented in the browser.
    app.run(debug=False)