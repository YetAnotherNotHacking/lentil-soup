# Minecraft Q-Learning Experience Reproducing Nureal Network, MQERNN
This is a script that was developed for fun to test how well machine learning would
work on the new system of the Mineflayer network api in bot.js. The api has several 
different points of data do be collected, which are inputted into the neural network, 
and then it is asked to choose a task, and that task is then carried out.

The bot.js just runs a server, and the python logic runs the machine learning tasks
and handle creating and saving the model, and all other logic. Note that running this
on a public network is a terrible idea, as the API is not yet secured, anyone can log
in from it, and anyone is able to get information from it and use the methods of moving
and everything without any real authentication. This is most likely not going to change.
Make sure you set your firewall correctly. Also important to note that the model file is
4.2-ish GB.

You can see at the top of the script where you are able to input the email, server ip,
port, and authentication method, microsoft for online servers, and offline for cracked.

Feel free to join the server sf-host-test.play.minekube.net:25565 where the bot is
commonly being tested and trained on. It is cracked, no need to worry.

# Installation

Install node for your operating system

Install required packages for bot.js

```
npm install mineflayer mineflayer-pathfinder express
```

Install required packages for python

```
pip3 install pytorch numpy matplotlib
```

# Credits:
Main developer, YetAnotherNotHacking

Training Assistant/Debugger - therpro (https://github.com/therpro)
