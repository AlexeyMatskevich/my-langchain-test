import { ChatWebLLM } from "@langchain/community/chat_models/webllm";
import { HumanMessage } from "@langchain/core/messages";
import { PGlite } from '@electric-sql/pglite';
import { live } from '@electric-sql/pglite/live'
import { vector } from '@electric-sql/pglite/vector'
import { PGLiteVectorStore } from "./pglitevector";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
// import { v4 as uuidv4 } from "uuid";
import { END, START, Annotation, StateGraph } from "@langchain/langgraph/web";
import { pull } from "langchain/hub";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import type { Document } from "@langchain/core/documents";

const embeddings = new HuggingFaceTransformersEmbeddings({
    model: "nomic-ai/nomic-embed-text-v1.5",
});

const pgClient = await PGlite.create({
    extensions: {
        live,
        vector,
    },
})

const config = {
    pgClient: pgClient,
    tableName: "testlangchainjs",
    columns: {
        idColumnName: "id",
        vectorColumnName: "vector",
        contentColumnName: "content",
        metadataColumnName: "metadata",
    }
};

const vectorStore = await PGLiteVectorStore.initialize(embeddings, config);

// const document1: Document = {
//     pageContent: "The powerhouse of the cell is the mitochondria",
//     metadata: { source: "https://example.com" },
// };
//
// const document2: Document = {
//     pageContent: "Buildings are made out of brick",
//     metadata: { source: "https://example.com" },
// };
//
// const document3: Document = {
//     pageContent: "Mitochondria are made out of lipids",
//     metadata: { source: "https://example.com" },
// };
//
// const document4: Document = {
//     pageContent: "The 2024 Olympics are in Paris",
//     metadata: { source: "https://example.com" },
// };
//
// const documents = [document1, document2, document3, document4];
//
// const ids = [uuidv4(), uuidv4(), uuidv4(), uuidv4()];

// await vectorStore.addDocuments(documents, { ids: ids });

// const filter = { source: "https://example.com" };
//
// const similaritySearchResults = await vectorStore.similaritySearch(
//     "biology",
//     2,
//     filter
// );
//
// for (const doc of similaritySearchResults) {
//     console.log(`* ${doc.pageContent} [${JSON.stringify(doc.metadata, null)}]`);
// }

window.pgClient = pgClient;

await pgClient.exec(`
  CREATE TABLE IF NOT EXISTS todo (
    id SERIAL PRIMARY KEY,
    task TEXT,
    done BOOLEAN DEFAULT false
  );
`)

const liveChanges = await pgClient.live.changes(
    "SELECT id, task, done FROM todo ORDER BY id;",
    [],           // параметры запроса
    "id",         // primary key для diff
    async (changes) => {  // callback входит массив изменений
        let newDocuments: Document[] = [];
        let updatedDocuments: Document[] = [];

        for (const c of changes) {
            switch (c.__op__) {
                case "INSERT":
                    console.log("Вставлена строка:", c);
                    newDocuments.push({
                        pageContent: c.task,
                        metadata: { done: c.done, id: c.id},
                    });
                    break;
                case "UPDATE":
                    console.log("Обновлена строка:", c);
                    updatedDocuments.push({
                        pageContent: c.task,
                        metadata: { done: c.done, id: c.id},
                    });
                    break;
                case "DELETE":
                    console.log("Удалена строка:", c);
                    break;
            }
        }
        // Теперь await будет работать
        if (newDocuments.length > 0) {
            console.log("Добавлено документов:", newDocuments.length);
            await vectorStore.addDocuments(newDocuments);
        }
        if (updatedDocuments.length > 0) {
            console.log("Обновлено документов:", updatedDocuments.length);
            await vectorStore.addDocuments(updatedDocuments);
        }
    }
);

await pgClient.exec(`
    INSERT INTO todo (task, done) VALUES ('Install PGlite from NPM', true);
    INSERT INTO todo (task, done) VALUES ('Load PGlite', true);
    INSERT INTO todo (task, done) VALUES ('Create a table', true);
    INSERT INTO todo (task, done) VALUES ('Insert some data', true);
    INSERT INTO todo (task) VALUES ('Update a task');
`)

// load model
const model = new ChatWebLLM({
    model: "Llama-3.2-1B-Instruct-q4f32_1-MLC",
    chatOptions: {
        temperature: 0.5,
    },
});

await model.initialize((progress) => {
    console.log(progress);
});

// Define prompt for question-answering
const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt");

// Define state for application
const InputStateAnnotation = Annotation.Root({
    question: Annotation<string>,
});
const StateAnnotation = Annotation.Root({
    question: Annotation<string>,
    context: Annotation<Document[]>,
    answer: Annotation<string>,
});

// Define application steps
const retrieve = async (state: typeof InputStateAnnotation.State) => {
    const retrievedDocs = await vectorStore.similaritySearch(state.question)
    return { context: retrievedDocs };
};


const generate = async (state: typeof StateAnnotation.State) => {
    const docsContent = state.context.map(doc => doc.pageContent).join("\n");
    const messages = await promptTemplate.invoke({ question: state.question, context: docsContent });
    const response = await model.invoke(messages);
    return { answer: response.content };
};


// Compile application and test
const graph = new StateGraph(StateAnnotation)
    .addNode("retrieve", retrieve)
    .addNode("generate", generate)
    .addEdge(START, "retrieve")
    .addEdge("retrieve", "generate")
    .addEdge("generate", END)
    .compile();

window.graph = graph;
//
// window.model = model;
// window.HumanMessage = HumanMessage;