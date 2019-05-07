# Conditional cartoon faces incorporating a landmark loss in cycleGAN
We learn how to map a face to a cartoon-face only using unpaired data. For a certain face the mapping to a cartoon style is a complex function ![functionG](doc/images/function-g.png). The goal of this project is to learn this function! As starting point we choose CycleGAN.
CycleGAN promises to learn the mapping from
![functionG](doc/images/function-g.png) and its reverse ![functionH](doc/images/function-h.png)
using unpaired data. The main idea is to minimize the loss of the transitive "cycle" ![eq](doc/images/transitive.png).
As well as a standard GAN Discriminator losses of each mapping function.

Our contributions:

In order to incorporate additional prior knowledge to the architecture we add another loss term based on landmark predictions. For both: the cartoons and the real images.

To extend the usages of our model we provide a one to ten mapping of real faces to cartoon images: We make the whole architecutre conditional and train the models with ten different hair colors. You are free to choose your color!

## Introduction
Cartoons have their own field in illustration and every artist creates his own style and strokes. Once a style is settled, many artist are able to copy and adapt the patterns and exaggerations. Faces, on the other hand, are hard to draw and also to adapt in style: the right proportions and asymmetries have to match each other perfectly.

The aim of this project is to automatically create faces in the style of the artist Shiraz Fuman. Our creations can then be used as starting point for your own face cartoon or used directly in large scale.

The CycleGAN approach respects the overall structure of the transformed input image: the face will have the same proportions and boundaries like the natural capture. This brings liveliness to drawings.

![demo](doc/images/cyclegan-demo.jpg)

Using a CycleGAN 

We plan to train a [CycleGAN](https://junyanz.github.io/CycleGAN/) on cartoon faces.

## DCGAN
We trained a vanilla DCGAN on the cartoon dataset (10k) in order to learn
how to generate women with beards:

![dcgan](doc/images/dcgan-fake-sample.png)

## Git Workflow
Just use classic merge commits if you find out that someone has pushed in meantime.

- git add .
- git commit -m 'message'
- git pull
- [do the merge, git add file, git commit]
- git push

## Project plan
To edit use: https://www.tablesgenerator.com/markdown_tables

| Week 	| Suggestion                	| Planned                                          	| Done 	|
|------	|---------------------------	|--------------------------------------------------	|------	|
| 1    	| create groups             	|                                                  	|      	|
| 2    	| identify challenges       	|                                                  	|      	|
| 3    	| read literature           	|                                                  	|      	|
| 4    	| propose solution          	|                                                  	|      	|
| 5    	| propose solution          	| literature, search for data, project description 	| yes  	|
| 6    	| propose solution (due A1) 	| download data, preprocessing ides (not much)      | yes   |
| 7    	| data preparation          	| everybody runs cycleGAN                         	| some  |
| 8    	| coding                    	| train on cropped faces, start implementation    	|      	|
| 9    	| coding (due A2)           	|                                                  	|      	|
| 10   	| easter holiday            	|                                                  	|      	|
| 11   	| net training              	|                                                  	|      	|
| 12   	| net training              	|                                                  	|      	|
| 13   	| experiments               	|                                                  	|      	|
| 14   	| Report & Presentation     	|                                                  	|      	|
| 15   	| Review                    	|                                                  	|      	|

## Issues

We define the faces as domain A, the cartoons as domain B. We store the images in folders under:
./datasets/customName/{trainA, trainB, testA, testB}. The results under ./results/customName/.

To work on a task, put your name in the column and shout to slack!


| Task                    	| Description                                                                                                                    	| Assigned 	| Status 	|
|-------------------------	|--------------------------------------------------------------------------------------------------------------------------------	|----------	|--------	|
| Parse command line args 	| Provide a way to store and retrieve configurations.                                                                            	|          	|        	|
| DataLoader              	| Load and preprocess (crop) images out of the folders trainA, trainB, testA, testB                                              	| Jan      	|        	|
| CycleGAN Training       	| - Create some first architectures for the Generator{A2B, B2A} and Discriminator{A, B} Networks. - Implement training procedure 	| Gautam   	| Working on |
| CycleGAN Test           	| - Implement test procedure "run GeneratorA2B"                                                                                  	|          	|        	|
| Training Visualization  	| - Dump learning rates (per Network) - Dump network weights - visualize reconstruction "A to B to A"                            	|          	|        	|
| Test Visualization      	| have fun ...                                                                                                                   	|          	|        	|

## Markdown O.o
Help is here: https://guides.github.com/features/mastering-markdown/


