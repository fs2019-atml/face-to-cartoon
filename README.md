# face-to-cartoon
We learn how to map a face to a cartoon-face only using unpaired data. For a certain face the mapping to a cartoon style is a complex function ![functionG](doc/images/function-g.png). The goal of this project is to learn this function! As starting point we choose CycleGAN.
CycleGAN promises to learn the mapping from
![functionG](doc/images/function-g.png) and its reverse ![functionH](doc/images/function-h.png)
using unpaired data. The main idea is to minimize the loss of the transitive "cycle" ![eq](doc/images/transitive.png).
As well as two standard GAN losses of each mapping function.

Using the CycleGAN as baseline we will investigate on more techniques and tricks to improve the results.

## Project description
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
| 6    	| propose solution (due A1) 	| download data, implement Prototype, preprocessing |    	  |
| 7    	| data preparation          	|                                                  	|      	|
| 8    	| coding                    	|                                                  	|      	|
| 9    	| coding (due A2)           	|                                                  	|      	|
| 10   	| easter holiday            	|                                                  	|      	|
| 11   	| net training              	|                                                  	|      	|
| 12   	| net training              	|                                                  	|      	|
| 13   	| experiments               	|                                                  	|      	|
| 14   	| Report & Presentation     	|                                                  	|      	|
| 15   	| Review                    	|                                                  	|      	|

## Markdown O.o
Help is here: https://guides.github.com/features/mastering-markdown/


