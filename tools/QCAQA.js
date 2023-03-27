const { Tool } = require('langchain/tools');
const { PineconeClient } = require('@pinecone-database/pinecone');
const { PineconeStore } = require('langchain/vectorstores');
const { OpenAIEmbeddings } = require('langchain/embeddings');
const { OpenAI } = require('langchain/llms');
const { VectorDBQAChain } = require('langchain/chains');

class QCAQA extends Tool {
    constructor() {
        super();
        this.name = 'qca-doc-qa';
        this.description = 'useful when query wifi driver information from QCA document.';

        this.run();
    }

    async run() {
        const client = new PineconeClient();
        await client.init({
            apiKey: process.env.PINECONE_API_KEY,
            environment: process.env.PINECONE_ENVIRONMENT
        });
        const pineconeIndex = client.Index(process.env.PINECONE_INDEX);
        console.log('Pinecone index is ready to use.');

        const vectorStore = await PineconeStore.fromExistingIndex(
            pineconeIndex,
            new OpenAIEmbeddings()
        );

        /* Use as part of a chain (currently no metadata filters) */
        const model = new OpenAI({ temperature: 0.2 });
        const chain = VectorDBQAChain.fromLLM(model, vectorStore, {
            k: 10,
            returnSourceDocuments: true
        });
        this.chain = chain;
        console.log('QCAQA is ready to use.');
    }

    async call(input) {
        const response = await this.chain.call({ query: input });
        console.log(response);
        return response;
    }
}

module.exports.QCAQA = QCAQA;
