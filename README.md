# face-to-cartoon
We learn how to map a face to a cartoon-face only using unpaired data. For a certain face the mapping to a cartoon style version of it is a complex function. The goal of this project is to learn this function! As starting point we choose CycleGAN:
Given unpaired data (just a set of faces and a set of cartoon-faces) CycleGAN promises to learn the mapping from
![functionG](doc/images/function-g.png)
and ![functionH](doc/images/function-h.png)
by minimizing the loss of the transitive "cycle" ![eq](doc/images/transitive.png)
Next to two standard GAN losses of each mapping function.

Using the CycleGAN as baseline we will investigate on more techniques and tricks to improve the results.

## Project description
We plan to train a [CycleGAN](https://junyanz.github.io/CycleGAN/) on cartoon faces.

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
| 5    	| propose solution          	| Literature, Data collection, Project description 	|      	|
| 6    	| propose solution (due A1) 	|                                                  	|      	|
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


