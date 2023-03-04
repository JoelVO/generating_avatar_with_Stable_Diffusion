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


**ADD IMAGES**

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
so taking it as a sign of the latent space not being as nice for that word, I created prompts for "woman" and "man", using the former to obtain $\omega$
and testing it on both sets.

**ADD IMAGES**


At this point you may be asking a very valid question: why don't generate an ordered batch in which the a person of age $a_1$ is in different situations 
and another batch with the same order in which the person has another age $a_2$ in the same situations as before, obtain the parwise difference
between corresponding elements and take the interpolation vector as its mean? 
Well, I have two answers for that. The first reason is because of PCA's ability for deal better while removing noise. You can see below the outcome of
applying the previously pca-obtained interpolation vector to the promp "A happy 20 years old man" vs. the one obtained by getting the mean.

**ADD IMAGES**

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

**ADD IMAGES**

But there may be even harder cases in which we are working with a characteristic with only few classes, for example having short, medium or long hair
or wearing glasses or not. Since I'm using PCA, applying it to a two prompts dataset ("person short/medium/long hair") doesn't make too much sense,
but I've mentioned that obtaining averaging creates a lot of undesired changes on our images. Because of this and since I can obtain more than 
just one principal component, I created prompts in which I was not only changing the description of the hair length but also on the place the person
was, which gave me the vector to create the following images.


**ADD IMAGES**


Before we go on, I would like to mention that this also talks about the way Stable Diffusion encodes texts and how it recognizes some
properties as continuous and some others as binary despite I provided more possible categories.


**ADD IMAGES**

Just to prove this was not only something from the human-related encodings, I tested the same technique to make the hours or months pass in 
a Central Park's image. The method proved to be successful and I generated some gifs to show how it not only distinguishes hours from months 
but also how smooth the transition is.

| Hours passing by at Central Park  | Months passing by at Central Park |
| ------------- | ------------- |
| ![landscape_hours](https://user-images.githubusercontent.com/57953211/222930196-65d9e934-2b89-4db7-84e4-a5073caf93b7.gif) | ![landscape_months](https://user-images.githubusercontent.com/57953211/222930203-7f84f3bc-2ce8-40b2-8498-464db5ba62cc.gif)
  |


