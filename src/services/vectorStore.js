const fs = require('fs');
const path = require('path');

let pipeline;

async function getEmbedder() {
    if (!pipeline) {
        const { pipeline: createPipeline } = await import('@xenova/transformers');
        pipeline = await createPipeline(
            'feature-extraction',
            'Xenova/all-MiniLM-L6-v2'  // runs locally, no API needed
        );
    }
    return pipeline;
}

class VectorStore {
    constructor() {
        this.storeBasePath = path.join(process.cwd(), 'data');
        this.stores = {
            erp: { documents: [], path: path.join(process.cwd(), 'data', 'vector_store_erp.json') },
            hrms: { documents: [], path: path.join(process.cwd(), 'data', 'vector_store_hrms.json') }
        };
    }

    async initialize() {
        if (!fs.existsSync(this.storeBasePath)) {
            fs.mkdirSync(this.storeBasePath, { recursive: true });
        }
        console.log('Loading embedding model...');
        await getEmbedder(); // preload on startup
        console.log('Embedding model ready');
    }

    async generateEmbedding(text) {
        try {
            const embedder = await getEmbedder();
            const output = await embedder(text, {
                pooling: 'mean',
                normalize: true
            });
            return Array.from(output.data);
        } catch (error) {
            console.error('Embedding error:', error.message);
            throw error;
        }
    }

    cosineSimilarity(vecA, vecB) {
        const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
        const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
        const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
        return dotProduct / (magnitudeA * magnitudeB);
    }

    async addDocuments(chunks, documentType) {
        if (!this.stores[documentType]) {
            throw new Error(`Invalid document type: ${documentType}`);
        }

        console.log(`Generating embeddings for ${documentType.toUpperCase()}...`);
        this.stores[documentType].documents = [];

        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            console.log(`Processing chunk ${i + 1}/${chunks.length}...`);

            const embedding = await this.generateEmbedding(chunk);

            this.stores[documentType].documents.push({
                id: `${documentType}_doc_${i}`,
                text: chunk,
                embedding,
                metadata: {
                    chunk_id: i,
                    source: `${documentType}_guide.pdf`,
                    documentType
                }
            });
        }

        fs.writeFileSync(
            this.stores[documentType].path,
            JSON.stringify(this.stores[documentType].documents, null, 2)
        );

        console.log(`${chunks.length} ${documentType.toUpperCase()} documents stored`);
    }

    async search(query, documentType, nResults = 3) {
        if (!this.stores[documentType]) {
            throw new Error(`Invalid document type: ${documentType}`);
        }

        if (this.stores[documentType].documents.length === 0 &&
            fs.existsSync(this.stores[documentType].path)) {
            const data = fs.readFileSync(this.stores[documentType].path, 'utf-8');
            this.stores[documentType].documents = JSON.parse(data);
        }

        if (this.stores[documentType].documents.length === 0) {
            throw new Error(`No documents in ${documentType.toUpperCase()} vector store. Please train first.`);
        }

        const queryEmbedding = await this.generateEmbedding(query);

        const similarities = this.stores[documentType].documents.map(doc => ({
            text: doc.text,
            similarity: this.cosineSimilarity(queryEmbedding, doc.embedding)
        }));

        similarities.sort((a, b) => b.similarity - a.similarity);
        const top = similarities.slice(0, nResults);

        console.log(`Top similarity: ${top[0]?.similarity.toFixed(3)}`);
        return top.map(r => r.text);
    }

    isTrained(documentType) {
        return fs.existsSync(this.stores[documentType]?.path || '');
    }

    async load(documentType) {
        if (fs.existsSync(this.stores[documentType].path)) {
            const data = fs.readFileSync(this.stores[documentType].path, 'utf-8');
            this.stores[documentType].documents = JSON.parse(data);
            console.log(`Loaded ${this.stores[documentType].documents.length} ${documentType.toUpperCase()} docs`);
            return true;
        }
        return false;
    }

    getAvailableDocuments() {
        return Object.keys(this.stores).filter(type => this.isTrained(type));
    }
}

module.exports = new VectorStore();