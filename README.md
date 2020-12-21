# Introduction 
Face out landmark features extractor.
Extracts the presence of Hijab, Niqab, hat and beard and detects hair color as well.

# Getting Started
1.	Clone the repo.
2.	run Dockerfile in order to import hair_color_image using
    docker build -t hair_color_image <location_of_docker_file>
3.	create a container based on CV_OUT_LANDMARK image
    docker run -d --name hair_color_container -p pn:5000  hair_color_image

# Usage

Use 192.168.100.44:pn/hair_color_api

using postman or any other framework send an image file with a post request to the mentioned endpoint
and you will get the response in the following format.

{
  "result": {
    "hair_color": {
      "label": [
        "brown",
        "blonde",
        "bald"
      ],
      "confidence": [
        100,
        100,
        99
      ]
    }
  }
}
