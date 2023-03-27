const { ChainTool } = require('langchain/tools');
const { VectorDBQAChain } = require('langchain/chains');
const { HNSWLib } = require('langchain/vectorstores');
const { OpenAIEmbeddings } = require('langchain/embeddings');
const fs = require('fs');
const csv = require('csv-parser');

class VectorQA extends ChainTool {
    constructor(model, name, description, filepath) {
        super({
            name: name || 'vector-db-qa',
            description: description || 'useful when the information is from embeddings.'
        });
        this.run(model, filepath);
    }

    async run(model, filepath) {
        const [texts, metas] = await this.extractCsvColumn(filepath, 'combined');
        console.log('11111\n');
        const vectorStore = await HNSWLib.fromTexts(
            texts,
            metas,
            new OpenAIEmbeddings());
        console.log('22222\n');
        const options = {
            returnSourceDocuments: true
        };
        const chain = VectorDBQAChain.fromLLM(
            model,
            vectorStore,
            options
        );
        this.chain = chain;
    }

    async extractCsvColumn(filePath, textColumn, sourceColumn) {
        return new Promise((resolve, reject) => {
            const texts = [];
            const metas = [];
            fs.createReadStream(filePath)
                .pipe(csv())
                .on('data', (data) => {
                    texts.push(data[textColumn]);
                    metas.push(data[sourceColumn]);
                })
                .on('end', () => {
                    resolve([texts, metas]);
                })
                .on('error', (error) => {
                    reject(error);
                });
        });
    }
}

module.exports.VectorQA = VectorQA;
