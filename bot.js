const { ActivityHandler, MessageFactory } = require('botbuilder');
const { OpenAI } = require('langchain');
const { initializeAgentExecutor } = require('langchain/agents');
const { Calculator } = require('langchain/tools');
const path = require('path');
const { BingSerpAPI } = require('./tools/BingSerpAPI');
const { DadJokeAPI } = require('./tools/DadJokeAPI');
const { PetFinderAPI } = require('./tools/FindPetsAPI');
const { IFTTTWebhook } = require('./tools/IFTTTWebhook');
const { QCAQA } = require('./tools/QCAQA');
const { VectorQA } = require('./tools/VectorQA');
const { BufferMemory } = require('langchain/memory');

class EchoBot extends ActivityHandler {
    constructor() {
        super();

        this.model = new OpenAI({ temperature: 0.9 });
        this.tools = [
            // new StateOfUnionQA(this.model, path.resolve(__dirname, './data/state_of_the_union.txt')),
            /* new VectorQA(
                this.model,
                'hitron-jira-qa',
                'hitron-jira-qa: handle the information queries for Hitron Products, such as HitronCloud.',
                path.resolve(__dirname, './data/Combined-JIRA.csv')),
            new BingSerpAPI(),
            new Calculator(),
            new DadJokeAPI(),
            new PetFinderAPI(),
            new IFTTTWebhook(
                `https://maker.ifttt.com/trigger/spotify/json/with/key/${process.env.IFTTTKey}`,
                'Spotify',
               'Play a song on Spotity.') 
            new QCAQA() */
            new QCAQA()
        ];

        this.onMessage(async (context, next) => {
            console.log('\n [BOT] Received Message');
            try {
                if (!this.executor) {
                    this.executor = await initializeAgentExecutor(
                        this.tools,
                        this.model,
                        'zero-shot-react-description'
                    );
                    console.log('Loaded agent.');
                }

                const input = context.activity.text;
                console.log('\n [BOT] Input:\n', input);
                const execResponse = await this.executor.call({input});
                console.log('\n [BOT] Response:\n', execResponse);

                const replyText = execResponse.output;

                // Print the log property of each action in intermediateSteps.
                // This is useful for debugging.
                execResponse.intermediateSteps.forEach((step) => {
                    console.log('----------[BOT]----------');
                    console.log(step.action.log);
                    console.log(`   [BOT]Observation: ${step.observation}`);
                });

                await context.sendActivity(MessageFactory.text(replyText, replyText));
            }
            catch (err) {
                console.log(err);
                throw err;
            }

            // By calling next() you ensure that the next BotHandler is run.
            await next();
        });


        this.onMembersAdded(async (context, next) => {
            const membersAdded = context.activity.membersAdded;
            const welcomeText = 'Hello and welcome!';
            for (let cnt = 0; cnt < membersAdded.length; ++cnt) {
                if (membersAdded[cnt].id !== context.activity.recipient.id) {
                    await context.sendActivity(MessageFactory.text(welcomeText, welcomeText));
                }
            }
            // By calling next() you ensure that the next BotHandler is run.
            await next();
        });
    }
}

module.exports.EchoBot = EchoBot;
