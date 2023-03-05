# Using Stable diffusion as a photo editor

## Motivation

Hey!

The original motivation behind doing this work was to make a little prank on whoever got my PhD positions application,
so if you're here because of that, I hope you didn't mind it, enjoyed it and have fun reading this in which I will try to show how this quickly became
a project I liked a lot and from which I learnt tons of new things. And if you just found this repository in the
internet sea, welcome too!
 
First of all, let me explain what the prank was. As I was planning to apply for PhD positions, I realized that a lot of researchers look for 
"creative and motivated" people to joing their teams, so I asked myself how can I show that and at the same time give a little taste of 
my abilities. With this in mind I had the idea of using a generated image as my cv picture and just mention at the end of it that the person
in the photo was not me, that it had been artificially produced and that if they were interested in how I did it, they can check out my github.

Nevertheless, I didn't have the time or resources to make a new generative model from scratch but most important, that wouldn't not creative at all.
We all have heard about Midjourney or played with Stable diffusion, so asking for a PhD position because I know how to search something in the internet
and input some text in an already trained network which wasn't mine seemed a bit unlikely to work, to say the least.

However, there had been some work before in which people have used generative networks to modify artificial images, such as 
in [FamilyGan](https://github.com/urielsinger/familyGan) where they used StyleGan to make predictions about how a couple's child would look like. I started reading about the topic and I got very surprised
to find out how people were finding vectors in the network's latent space which were allowing them to do all kind of stuff with people's faces
like changing their age, physiology or how big their smile is.

And that's how I had the idea for what I would do: I would take a picture of me and use one of this text-to-image generative models to see how
I could modify it so it would look like a photo I can put on my cv. 
It's worth mentioning it was never my intention to keep the resulting image to look like me because if you judge the result by that, you will
get disappointed and at the end of the day if I wanted that,I could just use my cellphone's camera. 
No, what I actually wanted to do was to see if I could change the purpose of this networks and use it to, given that I input a picture,
change it however I like.

So without further ado, let's get to it!

## Methodology

Choosing the network I would work with was an easy pick since keras already has a page dedicated to [Stable Diffusion](https://keras.io/examples/generative/random_walks_with_stable_diffusion/) and where it's shown how 
the conditional diffusion done with the prompts has a vector-space nature in the sense that, given that you use the same Gaussian distributed noise patch
and you enter two prompts,you can interpolate their encodded representation and the resulting images would be interpolated versions of your 
prompts themselves which is great because then I knew I will be working in a latent space where its vector space properties are more or less consistent.

That's great but, to which extent does this work? I mean, in their page Keras show this very cute example in which Golden Retriever becomes 
a bowl of fruit, so if I call the embedded versions of this prompts $p_1$ and $p_2$ respectively, then the interpolation process would be nothing
but $(1-t)p_1 + tp_2$ with $t$ going from 0 to 1. After rearranging, we have that we can understand the interpolation as 
$p_1 + t(p_2-p_1)$ which looks nice because if $p_2-p1$ has some meaning as *interpolator vector*, it would mean that other dogs can be turned into bowls of fruit just by 
adding that vector to the representation, maybe even some other animals.


Testing out this idea I took as my new prompts:

- prompt_3 = "A watercolor painting of a Grayhound at the beach"
- prompt_4 = "A Golden Retriever at the beach"
- prompt_5 = "A watercolor painting of a cat at the beach"

from which I got the following results using the interpolator vector I got before.


| prompt's image  | prompt plus interpolator vector|prompt's image  | prompt plus interpolator vector|
| ------------- | ------------- |------------- | ------------- |
|<p align="center"> prompt 1 </p>![or0](https://user-images.githubusercontent.com/57953211/222958270-a64be54d-1ff7-4d1b-93ad-5854c30df1b3.png)|<p align="center"> prompt 1 + interpolator </p>![or4](https://user-images.githubusercontent.com/57953211/222958273-72c3f5e9-1af0-405c-893e-39ac3cface10.png)|<p align="center"> prompt 3 </p>![0](https://user-images.githubusercontent.com/57953211/222958276-498be504-ccdc-4466-8fb3-63c8570d4cb2.png)|<p align="center"> prompt 3 + interpolator </p>![3](https://user-images.githubusercontent.com/57953211/222958279-079dd3c6-347d-4580-9a17-44d0086b7a5d.png)|
|<p align="center"> prompt 4 </p>![1](https://user-images.githubusercontent.com/57953211/222958277-5ad0fc99-d37a-46a4-b329-8c7a6d4410b5.png)|<p align="center"> prompt 4 + interpolator </p>![4](https://user-images.githubusercontent.com/57953211/222958280-dda7390c-c02f-442f-aeaf-1dbc719091f3.png)|<p align="center"> prompt 5 </p>![2](https://user-images.githubusercontent.com/57953211/222958278-6c93856b-84a7-4d91-831d-293b18ffe172.png)|<p align="center"> prompt 5 + interpolator </p>![5](https://user-images.githubusercontent.com/57953211/222958281-14b6df0f-25c3-4a18-b9a8-0561ddf43db5.png)|


It can be appreciated how despite still getting reasonable outputs, they have more modifications than expected since the bowl of fruit changed its background and content and in one case, the fruit even got wormholes. Nonetheless, this suggests interpolators can be generalized, but I would need to change the methodology in order to get what I'm looking for.

### Switching to humans


Now that I have a bit more idea of what I can do with this (although I haven't actually talked about modifiying a given picture), I questioned myself
which was the best way to obtain this interpolator vectors for, namely, make someone look older. It was clear that expecting a single instace to
be a good generalization I could use in a broader variety of cases was not going to work and that I had to incorporate the stochastic nature of it.

So let's assume $P$ and $I$ are the distributions of prompts discribing individual persons and of images showing a person in any situation respectively,
$f$ is the text encoder and $\Omega$ is Stable diffusion's latent space. Then, what I'm looking for is a vector $\omega \in \Omega$ such that
if $\pi_0$ is a prompt of any given person in some situation when they were $t_0$ years old and $\pi_1$ is a prompt of the exact same person 
in the exact same situation being $t_1$ years old, then $f(\pi_0) + (t_1-t_0)\omega = f(\pi_1)$ and thus $I(f(\pi_0) + (t_1-t_0)\omega)=I(f(\pi_1))$.

Hence, practically speaking and keeping the notation, one way to go further is to get a batch of prompts $x\sim P$ such that the context remains the
same, but what changes is the age of the person and apply PCA on $f[x]$, from which I will obtain $\omega$ as the principal component. I tried to try this
idea out with prompts literally just discribing a "person", but Stable diffusion proved to be quite bad at generating good images with them,
so taking it as a sign of the latent space not being as nice around that word, I created prompts for "woman" and "man", using the former to obtain $\omega$
and testing it on both sets.

| 20 years old  | 35 years old|50 years old|65 years old|80 years old|
| ------------- | ------------- |------------- | ------------- |------------- |
|![woman_0](https://user-images.githubusercontent.com/57953211/222959187-46f0b16b-cc9c-446a-94ba-c5ee3631ee70.png)|![woman_1](https://user-images.githubusercontent.com/57953211/222959188-77d9df08-ccb9-4c77-9fc6-30bbaca92697.png)|![woman_2](https://user-images.githubusercontent.com/57953211/222959189-a6ea4b2e-0cc1-44ff-8091-7a7c95ddf3ea.png)|![woman_3](https://user-images.githubusercontent.com/57953211/222959190-07526e5c-16de-4b53-be7f-9fe6b5af7d33.png)|![woman_4](https://user-images.githubusercontent.com/57953211/222959191-736e20e8-7ced-4718-8cd9-c4cf9475b98b.png)|
|![man_0](https://user-images.githubusercontent.com/57953211/222959387-6a02435e-9a91-4924-8ac9-41251452c978.png)|![man_1](https://user-images.githubusercontent.com/57953211/222959389-998fa624-c977-4dad-93d2-1dda130d9592.png)|![man_2](https://user-images.githubusercontent.com/57953211/222959390-198c5483-23a0-4e2e-9eed-bbab396ab2ef.png)|![man_3](https://user-images.githubusercontent.com/57953211/222959391-47dc4f29-4e53-480c-beaa-2cf5193c8e62.png)|![man_4](https://user-images.githubusercontent.com/57953211/222959392-eb13cced-105d-4f58-80d9-29b64bb67285.png)|



At this point you may be asking a very valid question: why don't generate an ordered batch in which the a person of age $a_1$ is in different situations 
and another batch with the same order in which the person has another age $a_2$ in the same situations as before, obtain the parwise difference
between corresponding elements and take the interpolation vector as its mean? 
Well, I have two answers for that. The first reason is because of PCA's ability for deal better while removing noise. You can see below the outcome of
applying the previously pca-obtained interpolation vector to the promp "A happy 20 years old man" vs. the one obtained by getting the mean.

|20 years old PCA|35 years old PCA|50 years old PCA|65 years old PCA|80 years old PCA|
| ------------- | ------------- |------------- | ------------- |------------- |
|![pca0](https://user-images.githubusercontent.com/57953211/222960288-0985d8db-0755-460c-bf8e-9f8a9e626848.png)|![pca1](https://user-images.githubusercontent.com/57953211/222960289-80950571-f03a-4640-a91c-a5ae94b0ec2f.png)|![pca2](https://user-images.githubusercontent.com/57953211/222960290-034671d9-05c5-4c9b-ac58-094e8c3bebb0.png)|![pca3](https://user-images.githubusercontent.com/57953211/222960291-fec4e5d5-3d0b-448d-9bed-37b5b18d9886.png)|![pca4](https://user-images.githubusercontent.com/57953211/222960292-16ebd4f2-b3c0-4ca9-a27a-e1247065413e.png)|




|20 years old difference vector|35 years old PCA difference vector|50 years old PCA difference vector|65 years old PCA difference vector|80 years old PCA difference vector|
| ------------- | ------------- |------------- | ------------- |------------- |
|![dif0](https://user-images.githubusercontent.com/57953211/222960321-24718cf5-ef1c-435b-aa11-45c8dd6afaac.png)|![dif1](https://user-images.githubusercontent.com/57953211/222960322-3cb1bad8-8e9e-4bb7-9d61-e582f8cb7478.png)|![dif2](https://user-images.githubusercontent.com/57953211/222960323-9e89bf5c-a7f0-4d73-8dc3-9f30c3b03160.png)|![dif3](https://user-images.githubusercontent.com/57953211/222960324-ae2010f0-53f9-414b-850a-d7b33ff2fee0.png)|![dif4](https://user-images.githubusercontent.com/57953211/222960325-ae93e51f-647c-4217-8e5b-a237bfc3342f.png)|



It can be appreciated how in both cases the person aged, but that in the mean version the person changed the shirt's position, got a hat or a wig,
got glasses and then lost them and changed the background, whereas for the one generated by the pca case, they kept the glasses once in the picture,
the shirt remained the same and the only change in the background was the appereance of a door.

The second reason I can offer to use pca over averaging is because we can obtain several principal components and this will come on handy later.


On another note, this last prompts were very easy to generate becase age is a characteristic we describe with a number, but what about the cases in which the feature is
not expressed this way? For this I generated promp batches, this time of men and women wearing shirts of different colors, again obtaining 
my interpolator vector from the women dataset and testing it in men's and in a new batch of women wearing skirts.

It may be worth mentioning that even if colors can be represented as numbers, we are working with a network trained on natural language and
I cannot recall the last time I heard somebody praising the 570 to 590 nm wavelength in van Gogh's paintings or asking for a B60017 apple and thus the not 
only discrete nature of the colors but also the impossibility of embedding them continuously into $\mathbb{N}$.

|blue skirt|1st color change|2nd color change|3rd color change|gray skirt|
| ------------- | ------------- |------------- | ------------- |------------- |
|![skirt0](https://user-images.githubusercontent.com/57953211/222960585-e9e451b4-cf2a-4a12-90c1-c420d92da516.png)|![skirt1](https://user-images.githubusercontent.com/57953211/222960586-acc8d313-a950-41dc-b58d-50353973aa9f.png)|![skirt2](https://user-images.githubusercontent.com/57953211/222960587-60470052-d448-41e9-94bf-c16d65a86dc6.png)|![skirt3](https://user-images.githubusercontent.com/57953211/222960588-cd8c7eeb-b10f-444d-a2bf-9d3abcc435db.png)|![skirt4](https://user-images.githubusercontent.com/57953211/222960589-90180184-9101-4dbc-9e87-5c9633d00f07.png)|



But there may be even harder cases in which we are working with a characteristic with only few classes, for example having short, medium or long hair
or wearing glasses or not. Since I'm using PCA, applying it to a two prompts dataset ("person short/medium/long hair") doesn't make too much sense,
but I've mentioned that obtaining averaging creates a lot of undesired changes on our images. Because of this and since I can obtain more than 
just one principal component, I created prompts in which I was not only changing the description of the hair length but also on the place the person
was, which gave me the vector to create the following images.

|short hair|long hair|even longer hair|
| ------------- | ------------- |------------- |
|![whairs](https://user-images.githubusercontent.com/57953211/222960924-2beb725c-d506-4c8e-834f-7aea0277c1a5.png)|![whairm](https://user-images.githubusercontent.com/57953211/222960927-de85c242-049f-48b1-b93a-6f383c7e84f1.png)|![whairl](https://user-images.githubusercontent.com/57953211/222960929-4b833ad7-beb7-4ba8-a1f0-2f019709997d.png)|

|short hair|medium hair|long hair|
| ------------- | ------------- |------------- |
|![mhairs](https://user-images.githubusercontent.com/57953211/222960960-fa8abdd4-17f1-43d3-aedf-5772413a2246.png)|![mhairm](https://user-images.githubusercontent.com/57953211/222960961-156ec811-8df2-481f-aa59-e9d85bd383ab.png)|![mhairl](https://user-images.githubusercontent.com/57953211/222960962-755ae1bf-e011-4307-a43f-22fdd59e34b9.png)|



Before we go on, let me say something about an extra meaning we can assign to the principal component vector I'm using. Since I was getting the principal component $v$ and was using it to move around the latent space, I could build a function $\phi$ from the latent space to $\mathbb{R}$ such that $\phi(x) = x\cdot v$. Thus, if $c\in\mathbb{R}$ is the parameter which will parametrize the movement on the direction $v$, $\phi(x+cv) = x\cdot v + c||v||^2 = c\cdot v+c$ since $v$ is normalized. Hence, we can understand $c$ as the amount of units we'll be moving in the desired direction, i.e. how many years the person will age, how many color units the clothing will change or how many length units the hair will grow. It also give us some insight into the way Stable Diffusion understand the properties we are talking about.


**ADD IMAGES**

Just to prove this was not only something from the human-related encodings, I tested the same technique to make the hours or months pass in 
a Central Park's image. The method proved to be successful and I generated some gifs to show how it not only distinguishes hours from months 
but also how smooth the transition is.

| Hours passing by at Central Park  | Months passing by at Central Park |
| ------------- | ------------- |
| ![landscape_hours](https://user-images.githubusercontent.com/57953211/222930196-65d9e934-2b89-4db7-84e4-a5073caf93b7.gif) | ![landscape_months](https://user-images.githubusercontent.com/57953211/222930203-7f84f3bc-2ce8-40b2-8498-464db5ba62cc.gif)
  |


