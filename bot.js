const express = require('express');
const mineflayer = require('mineflayer');
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder');
const Vec3 = require('vec3');

const app = express();
const port = 3000;

let bot = null;

app.use(express.json());

app.post('/login', (req, res) => {
  if (bot) {
    return res.status(400).send('Bot is already logged in.');
  }

  const { address, port: serverPort, account, auth } = req.body;

  bot = mineflayer.createBot({
    host: address,
    port: serverPort,
    username: account,
    auth: auth
  });

  bot.loadPlugin(pathfinder);

  bot.once('spawn', () => {
    const movements = new Movements(bot);
    bot.pathfinder.setMovements(movements);
    res.send('Bot logged in successfully.');
  });

  bot.on('error', (err) => {
    console.error(`Bot error: ${err.message}`);
    res.status(500).send(`Error: ${err.message}`);
  });

  bot.on('end', () => {
    bot = null;
  });
});

app.post('/move', (req, res) => {
  if (!bot) {
    return res.status(400).send('Bot is not logged in.');
  }

  const { x, y, z } = req.body;
  const goal = new goals.GoalBlock(x, y, z);

  bot.pathfinder.setGoal(goal, true)
    .then(() => res.send('Moved successfully.'))
    .catch(err => {
      console.error(`Error moving: ${err.message}`);
      res.status(500).send(`Error: ${err.message}`);
    });
});

app.post('/look', (req, res) => {
  if (!bot) {
    return res.status(400).send('Bot is not logged in.');
  }

  const { yaw, pitch } = req.body;

  bot.look(yaw, pitch)
    .then(() => res.send('Looked successfully.'))
    .catch(err => {
      console.error(`Error looking: ${err.message}`);
      res.status(500).send(`Error: ${err.message}`);
    });
});

app.post('/hit', (req, res) => {
  if (!bot) {
    return res.status(400).send('Bot is not logged in.');
  }

  const { x, y, z } = req.body;
  const targetBlock = bot.blockAt(new Vec3(x, y, z));

  bot.attack(targetBlock)
    .then(() => res.send('Attacked successfully.'))
    .catch(err => {
      console.error(`Error attacking: ${err.message}`);
      res.status(500).send(`Error: ${err.message}`);
    });
});

app.post('/mine', (req, res) => {
  if (!bot) {
    return res.status(400).send('Bot is not logged in.');
  }

  const { x, y, z } = req.body;
  const block = bot.blockAt(new Vec3(x, y, z));

  bot.dig(block)
    .then(() => res.send('Mined successfully.'))
    .catch(err => {
      console.error(`Error mining: ${err.message}`);
      res.status(500).send(`Error: ${err.message}`);
    });
});

app.post('/use', (req, res) => {
  if (!bot) {
    return res.status(400).send('Bot is not logged in.');
  }

  const { x, y, z } = req.body;
  const block = bot.blockAt(new Vec3(x, y, z));

  bot.activateBlock(block)
    .then(() => res.send('Used block successfully.'))
    .catch(err => {
      console.error(`Error using block: ${err.message}`);
      res.status(500).send(`Error: ${err.message}`);
    });
});

app.get('/status', (req, res) => {
  if (!bot) {
    return res.status(400).send('Bot is not logged in.');
  }

  const nearbyPlayers = Object.keys(bot.players).map(playerName => {
    const player = bot.players[playerName];
    return {
      name: playerName,
      position: player?.entity?.position ?? 'Unknown'
    };
  });

  const mobs = Object.keys(bot.entities).map(entityId => {
    const entity = bot.entities[entityId];
    return {
      id: entityId,
      position: entity?.position ?? 'Unknown',
      type: entity?.name ?? 'Unknown'
    };
  });

  const status = {
    health: bot.health,
    position: bot.entity.position,
    rotation: {
      yaw: bot.entity.yaw,
      pitch: bot.entity.pitch
    },
    saturation: bot.food,
    nearbyPlayers: nearbyPlayers,
    mobs: mobs
  };

  res.json(status);
});

app.get('/world', (req, res) => {
  if (!bot) {
    return res.status(400).send('Bot is not logged in.');
  }

  const world = {
    dimension: bot.world?.dimension,
    seed: bot.world?.seed
  };

  res.json(world);
});

app.get('/block', (req, res) => {
  if (!bot) {
    return res.status(400).send('Bot is not logged in.');
  }

  const { x, y, z } = req.query;
  const block = bot.blockAt(new Vec3(parseInt(x), parseInt(y), parseInt(z)));

  res.json({
    type: block.name,
    position: block.position
  });
});

app.get('/inventory', (req, res) => {
  if (!bot) {
    return res.status(400).send('Bot is not logged in.');
  }

  const inventory = bot.inventory.items().map(item => ({
    type: item.type,
    count: item.count,
    name: item.name
  }));

  res.json(inventory);
});

app.post('/chat', (req, res) => {
    if (!bot) {
        return res.status(400).send('Bot is not logged in.');
    }
    
    const { message } = req.body;
    
    bot.chat(message);
    res.send('Message sent.');
    }
);

app.post('/use-item', (req, res) => {
  if (!bot) {
    return res.status(400).send('Bot is not logged in.');
  }

  const { itemId, count } = req.body;

  const item = bot.inventory.items().find(i => i.type === itemId);
  if (item && item.count >= count) {
    bot.activateItem(item)
      .then(() => res.send('Item used successfully.'))
      .catch(err => {
        console.error(`Error using item: ${err.message}`);
        res.status(500).send(`Error: ${err.message}`);
      });
  } else {
    res.status(400).send('Item not found or insufficient quantity.');
  }
});

app.post('/equip-item', (req, res) => {
  if (!bot) {
    return res.status(400).send('Bot is not logged in.');
  }

  const { itemId } = req.body;

  const item = bot.inventory.items().find(i => i.type === itemId);
  if (item) {
    bot.equip(item, 'hand')
      .then(() => res.send('Item equipped successfully.'))
      .catch(err => {
        console.error(`Error equipping item: ${err.message}`);
        res.status(500).send(`Error: ${err.message}`);
      });
  } else {
    res.status(400).send('Item not found.');
  }
});

app.post('/logout', (req, res) => {
  if (bot) {
    bot.end();
    res.send('Control returned to host, no logic, disconnecting.');
    bot.chat('Control returned to host, no logic, disconnecting.');
  } else {
    res.status(400).send('Bot is not logged in.');
  }
});

app.post('/craft-inv', (req, res) => {
    if (!bot) {
        return res.status(400).send('Bot is not logged in.');
    }
    
    const { recipe } = req.body;
    const item = bot.recipesFor(recipe, null, 1)[0];
    
    if (item) {
        bot.craft(item, 1, null, (err) => {
        if (err) {
            console.error(`Error crafting: ${err.message}`);
            res.status(500).send(`Error: ${err.message}`);
        } else {
            res.send('Crafted successfully.');
        }
        });
    } else {
        res.status(400).send('Recipe not found.');
    }
    });

app.post('/craft-table', (req, res) => {
    if (!bot) {
        return res.status(400).send('Bot is not logged in.');
    }
    
    const { recipe, table } = req.body;
    const item = bot.recipesFor(recipe, table, 1)[0];
    
    if (item) {
        bot.craft(item, 1, table, (err) => {
        if (err) {
            console.error(`Error crafting: ${err.message}`);
            res.status(500).send(`Error: ${err.message}`);
        } else {
            res.send('Crafted successfully.');
        }
        });
    } else {
        res.status(400).send('Recipe not found.');
    }
    });

app.post('/open-chest', (req, res) => {
    if (!bot) {
        return res.status(400).send('Bot is not logged in.');
    }
    
    const { x, y, z } = req.body;
    const chest = bot.openChest(bot.blockAt(new Vec3(x, y, z)));
    
    if (chest) {
        res.send('Chest opened successfully.');
    } else {
        res.status(400).send('Chest not found.');
    }
    });

app.post('/deposit', (req, res) => {
    if (!bot) {
        return res.status(400).send('Bot is not logged in.');
    }
    
    const { x, y, z, itemName } = req.body;
    const chest = bot.blockAt(new Vec3(x, y, z));
    
    depositItems(bot, chest, itemName)
    .then(() => res.send('Items deposited successfully.'))
    .catch(err => {
        console.error(`Error depositing items: ${err.message}`);
        res.status(500).send(`Error: ${err.message}`);
    });
    });

app.post('/place-chest', (req, res) => {
    if (!bot) {
        return res.status(400).send('Bot is not logged in.');
    }
    
    const { x, y, z } = req.body;
    const chest = placeChest(bot, new Vec3(x, y, z));
    
    if (chest) {
        res.send('Chest placed successfully.');
    } else {
        res.status(400).send('Failed to place chest.');
    }
    });

app.post('/collect', (req, res) => {
    if (!bot) {
        return res.status(400).send('Bot is not logged in.');
    }
    
    const { itemName, requester } = req.body;
    
    collectItems(bot, itemName, requester)
    .then(() => res.send('Collection complete.'))
    .catch(err => {
        console.error(`Error collecting items: ${err.message}`);
        res.status(500).send(`Error: ${err.message}`);
    });
    });

app.post('/eat', (req, res) => {
    if (!bot) {
        return res.status(400).send('Bot is not logged in.');
    }
    
    bot.eat((err) => {
        if (err) {
            console.error(`Error eating: ${err.message}`);
            res.status(500).send(`Error: ${err.message}`);
        } else {
            res.send('Ate successfully.');
        }
    });
    }); 

app.post('/equip-item', (req, res) => {
    if (!bot) {
        return res.status(400).send('Bot is not logged in.');
    }
    
    const { itemId } = req.body;
    
    const item = bot.inventory.items().find(i => i.type === itemId);
    if (item) {
        bot.equip(item, 'hand')
        .then(() => res.send('Item equipped successfully.'))
        .catch(err => {
            console.error(`Error equipping item: ${err.message}`);
            res.status(500).send(`Error: ${err.message}`);
        });
    } else {
        res.status(400).send('Item not found.');
    }
    });

app.post('/drop', (req, res) => {
    if (!bot) {
        return res.status(400).send('Bot is not logged in.');
    }
    
    const { itemId, count } = req.body;
    
    const item = bot.inventory.items().find(i => i.type === itemId);
    if (item && item.count >= count) {
        bot.tossStack(item, count, (err) => {
        if (err) {
            console.error(`Error dropping item: ${err.message}`);
            res.status(500).send(`Error: ${err.message}`);
        } else {
            res.send('Item dropped successfully.');
        }
        });
    } else {
        res.status(400).send('Item not found or insufficient quantity.');
    }
    });

app.post('/smelt', (req, res) => {
    if (!bot) {
        return res.status(400).send('Bot is not logged in.');
    }
    
    const { itemId } = req.body;
    
    const item = bot.inventory.items().find(i => i.type === itemId);
    if (item) {
        bot.smelt(item, (err) => {
        if (err) {
            console.error(`Error smelting item: ${err.message}`);
            res.status(500).send(`Error: ${err.message}`);
        } else {
            res.send('Item smelted successfully.');
        }
        });
    } else {
        res.status(400).send('Item not found.');
    }
    });

app.post('attack', (req, res) => {
    if (!bot) {
        return res.status(400).send('Bot is not logged in.');
    }
    
    const { target } = req.body;
    
    const entity = bot.nearestEntity((entity) => entity.name === target);
    if (entity) {
        bot.attack(entity, true);
        res.send('Attacked successfully.');
    } else {
        res.status(400).send('Target not found.');
    }
    });



app.listen(port, () => {
    console.log(`API listening on port ${port}`);
}).on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
        console.error(`Port ${port} is already in use.`);
    } else {
        console.error(`An error occurred while starting the server: ${err.message}`);
    }
});
