from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

bot =ChatBot("chatbot",read_only=False,logic_adapters=["chatterbot.logic.BestMatch"])

list_trainer = ListTrainer(bot)

list_trainer.train()
