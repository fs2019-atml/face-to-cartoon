#!/bin/bash

#####
# downloads latex equations from http://www.sciweavers.org/free-online-latex-equation-editor to
#####

# get the images using curl
function get {
	curl -o $1 $2
}

# Images for readme.md
get function-g.png "http://www.sciweavers.org/tex2img.php?eq=G%3A%20X%20%5Crightarrow%20Y%0A&bc=White&fc=Black&im=png&fs=12&ff=arev&edit=0"
get function-h.png "http://www.sciweavers.org/tex2img.php?eq=H%3A%20Y%20%5Crightarrow%20X%0A&bc=White&fc=Black&im=png&fs=12&ff=arev&edit=0"
get transitive.png "http://www.sciweavers.org/tex2img.php?eq=x%20%3D%20H%28G%28x%29%29%20%0A&bc=White&fc=Black&im=png&fs=12&ff=arev&edit=0"
