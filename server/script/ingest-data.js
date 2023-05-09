import { UnstructuredLoader } from "langchain/document_loaders/fs/unstructured";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PineconeClient } from "@pinecone-database/pinecone";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import path from "path";
import * as dotenv from "dotenv";
import * as url from "url";
const __dirname = url.fileURLToPath(new URL(".", import.meta.url));

dotenv.config();

// const loader = new UnstructuredLoader(
//   path.resolve(__dirname, "../source/book.pdf")
// );
const loader = new PDFLoader(
   path.resolve(__dirname, "../source/book.pdf")
)

const rawDocs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const docs = await splitter.splitDocuments(rawDocs);
console.log(docs)

const embeddings = new OpenAIEmbeddings();

const client = new PineconeClient();
await client.init({
  apiKey: process.env.PINECONE_API_KEY,
  environment: process.env.PINECONE_ENVIRONMENT,
});

const pineconeIndex = client.Index(process.env.PINECONE_INDEX);

try {
  await PineconeStore.fromDocuments(docs, embeddings, {
    pineconeIndex,
    textKey: "text",
    namespace: "xyqxr",
  });
} catch (error) {
  console.log(error);
}
