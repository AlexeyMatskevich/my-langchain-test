import { PGlite, PGliteInterfaceExtensions } from '@electric-sql/pglite';
import {
    MaxMarginalRelevanceSearchOptions,
    VectorStore,
} from "@langchain/core/vectorstores";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import { Document } from "@langchain/core/documents";
import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { maximalMarginalRelevance } from "@langchain/core/utils/math";

type Metadata = Record<string, unknown>;

export type DistanceStrategy = "cosine" | "innerProduct" | "euclidean";

/**
 * Interface that defines the arguments required to create a
 * `PGLiteVectorStore` instance. It includes Postgres connection options,
 * table name, filter, and verbosity level.
 */
export interface PGLiteVectorStoreArgs {
    pgClient?: PGlite & PGliteInterfaceExtensions<any>;
    tableName: string;
    collectionTableName?: string;
    collectionName?: string;
    collectionMetadata?: Metadata | null;
    schemaName?: string | null;
    extensionSchemaName?: string | null;
    columns?: {
        idColumnName?: string;
        vectorColumnName?: string;
        contentColumnName?: string;
        metadataColumnName?: string;
    };
    filter?: Metadata;
    verbose?: boolean;
    /**
     * The number of documents to chunk by when
     * adding vectors.
     * @default 500
     */
    chunkSize?: number;
    ids?: string[];
    distanceStrategy?: DistanceStrategy;
}

export class PGLiteVectorStore extends VectorStore {
    declare FilterType: Metadata;

    tableName: string;

    collectionTableName?: string;

    collectionName = "langchain";

    collectionMetadata: Metadata | null;

    schemaName: string | null;

    idColumnName: string;

    vectorColumnName: string;

    contentColumnName: string;

    extensionSchemaName: string | null;

    metadataColumnName: string;

    filter?: Metadata;

    _verbose?: boolean;

    pgClient?: PGlite & PGliteInterfaceExtensions<any>;

    chunkSize = 500;

    distanceStrategy?: DistanceStrategy = "cosine";

    _vectorstoreType(): string {
        return "pgvector";
    }

    constructor(embeddings: EmbeddingsInterface, config: PGLiteVectorStoreArgs) {
        super(embeddings, config);
        this.tableName = config.tableName;
        if (
            config.collectionName !== undefined &&
            config.collectionTableName === undefined
        ) {
            throw new Error(
                `If supplying a "collectionName", you must also supply a "collectionTableName".`
            );
        }
        this.collectionTableName = config.collectionTableName;
        this.collectionName = config.collectionName ?? "langchain";
        this.collectionMetadata = config.collectionMetadata ?? null;
        this.schemaName = config.schemaName ?? null;
        this.extensionSchemaName = config.extensionSchemaName ?? null;

        this.filter = config.filter;

        this.vectorColumnName = config.columns?.vectorColumnName ?? "embedding";
        this.contentColumnName = config.columns?.contentColumnName ?? "text";
        this.idColumnName = config.columns?.idColumnName ?? "id";
        this.metadataColumnName = config.columns?.metadataColumnName ?? "metadata";

        if (!config.pgClient) {
            throw new Error(
                "You must provide a `PGlite` instance."
            );
        }
        this.pgClient = config.pgClient;
        this.chunkSize = config.chunkSize ?? 500;
        this.distanceStrategy = config.distanceStrategy ?? this.distanceStrategy;

        const langchainVerbose = getEnvironmentVariable("LANGCHAIN_VERBOSE");

        if (langchainVerbose === "true") {
            this._verbose = true;
        } else if (langchainVerbose === "false") {
            this._verbose = false;
        } else {
            this._verbose = config.verbose;
        }
    }

    get computedTableName() {
        return this.schemaName == null
            ? `${this.tableName}`
            : `"${this.schemaName}"."${this.tableName}"`;
    }

    get computedCollectionTableName() {
        return this.schemaName == null
            ? `${this.collectionTableName}`
            : `"${this.schemaName}"."${this.collectionTableName}"`;
    }

    get computedOperatorString() {
        let operator: string;
        switch (this.distanceStrategy) {
            case "cosine":
                operator = "<=>";
                break;
            case "innerProduct":
                operator = "<#>";
                break;
            case "euclidean":
                operator = "<->";
                break;
            default:
                throw new Error(`Unknown distance strategy: ${this.distanceStrategy}`);
        }

        return this.extensionSchemaName !== null
            ? `OPERATOR(${this.extensionSchemaName}.${operator})`
            : operator;
    }

    /**
     * Static method to create a new `PGLiteVectorStore` instance from a
     * connection. It creates a table if one does not exist, and calls
     * `connect` to return a new instance of `PGLiteVectorStore`.
     *
     * @param embeddings - Embeddings instance.
     * @param fields - `PGLiteVectorStoreArgs` instance
     * @param fields.dimensions Number of dimensions in your vector data type. For example, use 1536 for OpenAI's `text-embedding-3-small`. If not set, indexes like HNSW might not be used during query time.
     * @returns A new instance of `PGLiteVectorStore`.
     */
    static async initialize(
        embeddings: EmbeddingsInterface,
        config: PGLiteVectorStoreArgs & { dimensions?: number }
    ): Promise<PGLiteVectorStore> {
        const { dimensions, ...rest } = config;
        const postgresqlVectorStore = new PGLiteVectorStore(embeddings, rest);

        await postgresqlVectorStore.ensureTableInDatabase(dimensions);
        if (postgresqlVectorStore.collectionTableName) {
            await postgresqlVectorStore.ensureCollectionTableInDatabase();
        }

        return postgresqlVectorStore;
    }

    /**
     * Method to add documents to the vector store. It converts the documents into
     * vectors, and adds them to the store.
     *
     * @param documents - Array of `Document` instances.
     * @param options - Optional arguments for adding documents
     * @returns Promise that resolves when the documents have been added.
     */
    async addDocuments(
        documents: Document[],
        options?: { ids?: string[] }
    ): Promise<void> {
        const texts = documents.map(({ pageContent }) => pageContent);

        return this.addVectors(
            await this.embeddings.embedDocuments(texts),
            documents,
            options
        );
    }

    /**
     * Inserts a row for the collectionName provided at initialization if it does not
     * exist and returns the collectionId.
     *
     * @returns The collectionId for the given collectionName.
     */
    async getOrCreateCollection(): Promise<string> {
        const queryString = `
      SELECT uuid from ${this.computedCollectionTableName}
      WHERE name = $1;
    `;
        const queryResult = await this.pgClient.query<{ uuid: string }>(queryString, [
            this.collectionName,
        ]);
        let collectionId = queryResult.rows[0]?.uuid;

        if (!collectionId) {
            const insertString = `
        INSERT INTO ${this.computedCollectionTableName}(
          uuid,
          name,
          cmetadata
        )
        VALUES (
          gen_random_uuid(),
          $1,
          $2
        )
        RETURNING uuid;
      `;
            const insertResult = await this.pgClient.query<{ uuid: string }>(insertString, [
                this.collectionName,
                this.collectionMetadata,
            ]);
            collectionId = insertResult.rows[0]?.uuid;
        }

        return collectionId;
    }

    /**
     * Generates the SQL placeholders for a specific row at the provided index.
     *
     * @param index - The index of the row for which placeholders need to be generated.
     * @param numOfColumns - The number of columns we are inserting data into.
     * @returns The SQL placeholders for the row values.
     */
    private generatePlaceholderForRowAt(
        index: number,
        numOfColumns: number
    ): string {
        const placeholders = [];
        for (let i = 0; i < numOfColumns; i += 1) {
            placeholders.push(`$${index * numOfColumns + i + 1}`);
        }
        return `(${placeholders.join(", ")})`;
    }

    /**
     * Constructs the SQL query for inserting rows into the specified table.
     *
     * @param rows - The rows of data to be inserted, consisting of values and records.
     * @returns The complete SQL INSERT INTO query string.
     */
    private async buildInsertQuery(rows: (string | Record<string, unknown>)[][]) {
        let collectionId;
        if (this.collectionTableName) {
            collectionId = await this.getOrCreateCollection();
        }

        const columns = [
            this.contentColumnName,
            this.vectorColumnName,
            this.metadataColumnName,
        ];

        if (collectionId) {
            columns.push("collection_id");
        }

        // Check if we have added ids to the rows.
        if (rows.length !== 0 && columns.length === rows[0].length - 1) {
            columns.push(this.idColumnName);
        }

        const valuesPlaceholders = rows
            .map((_, j) => this.generatePlaceholderForRowAt(j, columns.length))
            .join(", ");

        const text = `
      INSERT INTO ${this.computedTableName}(
        ${columns.map((column) => `"${column}"`).join(", ")}
      )
      VALUES ${valuesPlaceholders}
    `;
        return text;
    }

    /**
     * Method to add vectors to the vector store. It converts the vectors into
     * rows and inserts them into the database.
     *
     * @param vectors - Array of vectors.
     * @param documents - Array of `Document` instances.
     * @param options - Optional arguments for adding documents
     * @returns Promise that resolves when the vectors have been added.
     */
    async addVectors(
        vectors: number[][],
        documents: Document[],
        options?: { ids?: string[] }
    ): Promise<void> {
        const ids = options?.ids;

        // Either all documents have ids or none of them do to avoid confusion.
        if (ids !== undefined && ids.length !== vectors.length) {
            throw new Error(
                "The number of ids must match the number of vectors provided."
            );
        }

        const rows = [];
        let collectionId;
        if (this.collectionTableName) {
            collectionId = await this.getOrCreateCollection();
        }

        for (let i = 0; i < vectors.length; i += 1) {
            const values = [];
            const embedding = vectors[i];
            const embeddingString = `[${embedding.join(",")}]`;
            values.push(
                documents[i].pageContent.replace(/\0/g, ""),
                embeddingString.replace(/\0/g, ""),
                documents[i].metadata
            );
            if (collectionId) {
                values.push(collectionId);
            }
            if (ids) {
                values.push(ids[i]);
            }
            rows.push(values);
        }

        for (let i = 0; i < rows.length; i += this.chunkSize) {
            const chunk = rows.slice(i, i + this.chunkSize);
            const insertQuery = await this.buildInsertQuery(chunk);
            const flatValues = chunk.flat();
            try {
                await this.pgClient.query(insertQuery, flatValues);
            } catch (e) {
                console.error(e);
                throw new Error(`Error inserting: ${(e as Error).message}`);
            }
        }
    }

    /**
     * Method to delete documents from the vector store. It deletes the
     * documents that match the provided ids.
     *
     * @param ids - Array of document ids.
     * @returns Promise that resolves when the documents have been deleted.
     */
    private async deleteById(ids: string[]) {
        let collectionId;
        if (this.collectionTableName) {
            collectionId = await this.getOrCreateCollection();
        }

        // Set parameters of dynamically generated query
        const params = collectionId ? [ids, collectionId] : [ids];

        const queryString = `
      DELETE FROM ${this.computedTableName}
      WHERE ${collectionId ? "collection_id = $2 AND " : ""}${
            this.idColumnName
        } = ANY($1::uuid[])
    `;
        await this.pgClient.query(queryString, params);
    }

    /**
     * Method to delete documents from the vector store. It deletes the
     * documents whose metadata contains the filter.
     *
     * @param filter - An object representing the Metadata filter.
     * @returns Promise that resolves when the documents have been deleted.
     */
    private async deleteByFilter(filter: Metadata) {
        let collectionId;
        if (this.collectionTableName) {
            collectionId = await this.getOrCreateCollection();
        }

        // Set parameters of dynamically generated query
        const params = collectionId ? [filter, collectionId] : [filter];

        const queryString = `
      DELETE FROM ${this.computedTableName}
      WHERE ${collectionId ? "collection_id = $2 AND " : ""}${
            this.metadataColumnName
        }::jsonb @> $1
    `;
        return await this.pgClient.query(queryString, params);
    }

    /**
     * Method to delete documents from the vector store. It deletes the
     * documents that match the provided ids or metadata filter. Matches ids
     * exactly and metadata filter according to postgres jsonb containment. Ids and filter
     * are mutually exclusive.
     *
     * @param params - Object containing either an array of ids or a metadata filter object.
     * @returns Promise that resolves when the documents have been deleted.
     * @throws Error if neither ids nor filter are provided, or if both are provided.
     * @example <caption>Delete by ids</caption>
     * await vectorStore.delete({ ids: ["id1", "id2"] });
     * @example <caption>Delete by filter</caption>
     * await vectorStore.delete({ filter: { a: 1, b: 2 } });
     */
    async delete(params: { ids?: string[]; filter?: Metadata }): Promise<void> {
        const { ids, filter } = params;

        if (!(ids || filter)) {
            throw new Error(
                "You must specify either ids or a filter when deleting documents."
            );
        }

        if (ids && filter) {
            throw new Error(
                "You cannot specify both ids and a filter when deleting documents."
            );
        }

        if (ids) {
            await this.deleteById(ids);
        } else if (filter) {
            await this.deleteByFilter(filter);
        }
    }

    /**
     * Method to perform a similarity search in the vector store. It returns the `k` most similar documents to the query text.
     * @param query - Query vector.
     * @param k - Number of most similar documents to return.
     * @param filter - Optional filter to apply to the search.
     * @param includeEmbedding Whether to include the embedding vectors in the results.
     * @returns Promise that resolves with an array of tuples, each containing a `Document` and its similarity score.
     */
    private async searchPostgres(
        query: number[],
        k: number,
        filter?: this["FilterType"],
        includeEmbedding?: boolean
    ): Promise<[Document, number][]> {
        const embeddingString = `[${query.join(",")}]`;
        const _filter: this["FilterType"] = filter ?? {};

        let collectionId;
        if (this.collectionTableName) {
            collectionId = await this.getOrCreateCollection();
        }

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const parameters: unknown[] = [embeddingString, k];
        const whereClauses = [];

        if (collectionId) {
            whereClauses.push("collection_id = $3");
            parameters.push(collectionId);
        }

        let paramCount = parameters.length;
        for (const [key, value] of Object.entries(_filter)) {
            if (typeof value === "object" && value !== null) {
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                const _value: Record<string, any> = value;
                const currentParamCount = paramCount;
                if (Array.isArray(_value.in)) {
                    const placeholders = _value.in
                        .map(
                            (_: unknown, index: number) => `$${currentParamCount + index + 1}`
                        )
                        .join(",");
                    whereClauses.push(
                        `${this.metadataColumnName}->>'${key}' IN (${placeholders})`
                    );
                    parameters.push(..._value.in);
                    paramCount += _value.in.length;
                }
                if (Array.isArray(_value.notIn)) {
                    const placeholders = _value.notIn
                        .map(
                            (_: unknown, index: number) => `$${currentParamCount + index + 1}`
                        )
                        .join(",");
                    whereClauses.push(
                        `${this.metadataColumnName}->>'${key}' NOT IN (${placeholders})`
                    );
                    parameters.push(..._value.notIn);
                    paramCount += _value.notIn.length;
                }
                if (Array.isArray(_value.arrayContains)) {
                    const placeholders = _value.arrayContains
                        .map(
                            (_: unknown, index: number) => `$${currentParamCount + index + 1}`
                        )
                        .join(",");
                    whereClauses.push(
                        `${this.metadataColumnName}->'${key}' ?| array[${placeholders}]`
                    );
                    parameters.push(..._value.arrayContains);
                    paramCount += _value.arrayContains.length;
                }
            } else {
                paramCount += 1;
                whereClauses.push(
                    `${this.metadataColumnName}->>'${key}' = $${paramCount}`
                );
                parameters.push(value);
            }
        }

        const whereClause = whereClauses.length
            ? `WHERE ${whereClauses.join(" AND ")}`
            : "";

        const queryString = `
      SELECT *, "${this.vectorColumnName}" ${this.computedOperatorString} $1 as "_distance"
      FROM ${this.computedTableName}
      ${whereClause}
      ORDER BY "_distance" ASC
      LIMIT $2;
      `;

        interface QueryResult {
            [key: string]: any;
            _distance: number;
        }
        const documents = (await this.pgClient.query<QueryResult>(queryString, parameters)).rows;

        const results = [] as [Document, number][];
        for (const doc of documents) {
            if (doc._distance != null && doc[this.contentColumnName] != null) {
                const document = new Document({
                    pageContent: doc[this.contentColumnName],
                    metadata: doc[this.metadataColumnName],
                    id: doc[this.idColumnName],
                });
                if (includeEmbedding) {
                    document.metadata[this.vectorColumnName] = doc[this.vectorColumnName];
                }
                results.push([document, doc._distance]);
            }
        }
        return results;
    }

    /**
     * Method to perform a similarity search in the vector store. It returns
     * the `k` most similar documents to the query vector, along with their
     * similarity scores.
     * @param query - Query vector.
     * @param k - Number of most similar documents to return.
     * @param filter - Optional filter to apply to the search.
     * @returns Promise that resolves with an array of tuples, each containing a `Document` and its similarity score.
     */
    async similaritySearchVectorWithScore(
        query: number[],
        k: number,
        filter?: this["FilterType"]
    ): Promise<[Document, number][]> {
        return this.searchPostgres(query, k, filter, false);
    }

    /**
     * Method to ensure the existence of the table in the database. It creates
     * the table if it does not already exist.
     * @param dimensions Number of dimensions in your vector data type. For example, use 1536 for OpenAI's `text-embedding-3-small`. If not set, indexes like HNSW might not be used during query time.
     * @returns Promise that resolves when the table has been ensured.
     */
    async ensureTableInDatabase(dimensions?: number): Promise<void> {
        const vectorQuery =
            this.extensionSchemaName == null
                ? "CREATE EXTENSION IF NOT EXISTS vector;"
                : `CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA "${this.extensionSchemaName}";`;
        const extensionName =
            this.extensionSchemaName == null
                ? "vector"
                : `"${this.extensionSchemaName}"."vector"`;
        const vectorColumnType = dimensions
            ? `${extensionName}(${dimensions})`
            : extensionName;
        const tableQuery = `
      CREATE TABLE IF NOT EXISTS ${this.computedTableName} (
        "${this.idColumnName}" uuid NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
        "${this.contentColumnName}" text,
        "${this.metadataColumnName}" jsonb,
        "${this.vectorColumnName}" ${vectorColumnType}
      );
    `;
        await this.pgClient.query(vectorQuery);
        await this.pgClient.query(tableQuery);
    }

    /**
     * Method to ensure the existence of the collection table in the database.
     * It creates the table if it does not already exist.
     *
     * @returns Promise that resolves when the collection table has been ensured.
     */
    async ensureCollectionTableInDatabase(): Promise<void> {
        try {
            const queryString = `
        CREATE TABLE IF NOT EXISTS ${this.computedCollectionTableName} (
          uuid uuid NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
          name character varying,
          cmetadata jsonb
        );

        CREATE INDEX IF NOT EXISTS idx_${this.collectionTableName}_name ON ${this.computedCollectionTableName}(name);

        ALTER TABLE ${this.computedTableName}
          ADD COLUMN collection_id uuid;

        ALTER TABLE ${this.computedTableName}
          ADD CONSTRAINT ${this.tableName}_collection_id_fkey
          FOREIGN KEY (collection_id)
          REFERENCES ${this.computedCollectionTableName}(uuid)
          ON DELETE CASCADE;
      `;
            await this.pgClient.query(queryString);
        } catch (e) {
            if (!(e as Error).message.includes("already exists")) {
                console.error(e);
                throw new Error(
                    `Error adding column or creating index: ${(e as Error).message}`
                );
            }
        }
    }

    /**
     * Static method to create a new `PGLiteVectorStore` instance from an
     * array of texts and their metadata. It converts the texts into
     * `Document` instances and adds them to the store.
     *
     * @param texts - Array of texts.
     * @param metadatas - Array of metadata objects or a single metadata object.
     * @param embeddings - Embeddings instance.
     * @param dbConfig - `PGLiteVectorStoreArgs` instance.
     * @returns Promise that resolves with a new instance of `PGLiteVectorStore`.
     */
    static async fromTexts(
        texts: string[],
        metadatas: object[] | object,
        embeddings: EmbeddingsInterface,
        dbConfig: PGLiteVectorStoreArgs & { dimensions?: number }
    ): Promise<PGLiteVectorStore> {
        const docs = [];
        for (let i = 0; i < texts.length; i += 1) {
            const metadata = Array.isArray(metadatas) ? metadatas[i] : metadatas;
            const newDoc = new Document({
                pageContent: texts[i],
                metadata,
            });
            docs.push(newDoc);
        }

        return PGLiteVectorStore.fromDocuments(docs, embeddings, dbConfig);
    }

    /**
     * Static method to create a new `PGLiteVectorStore` instance from an
     * array of `Document` instances. It adds the documents to the store.
     *
     * @param docs - Array of `Document` instances.
     * @param embeddings - Embeddings instance.
     * @param dbConfig - `PGLiteVectorStoreArgs` instance.
     * @returns Promise that resolves with a new instance of `PGLiteVectorStore`.
     */
    static async fromDocuments(
        docs: Document[],
        embeddings: EmbeddingsInterface,
        dbConfig: PGLiteVectorStoreArgs & { dimensions?: number }
    ): Promise<PGLiteVectorStore> {
        const instance = await PGLiteVectorStore.initialize(embeddings, dbConfig);
        await instance.addDocuments(docs, { ids: dbConfig.ids });

        return instance;
    }

    /**
     * Method to create the HNSW index on the vector column.
     *
     * @param dimensions - Defines the number of dimensions in your vector data type, up to 2000. For example, use 1536 for OpenAI's text-embedding-ada-002 and Amazon's amazon.titan-embed-text-v1 models.
     * @param m - The max number of connections per layer (16 by default). Index build time improves with smaller values, while higher values can speed up search queries.
     * @param efConstruction -  The size of the dynamic candidate list for constructing the graph (64 by default). A higher value can potentially improve the index quality at the cost of index build time.
     * @param distanceFunction -  The distance function name you want to use, is automatically selected based on the distanceStrategy.
     * @param namespace -  The namespace is used to create the index with a specific name. This is useful when you want to create multiple indexes on the same database schema (within the same schema in PostgreSQL, the index name must be unique across all tables).
     * @returns Promise that resolves with the query response of creating the index.
     */
    async createHnswIndex(config: {
        dimensions: number;
        m?: number;
        efConstruction?: number;
        distanceFunction?: string;
        namespace?: string;
    }): Promise<void> {
        let idxDistanceFunction = config?.distanceFunction || "vector_cosine_ops";
        const prefix = config?.namespace ? `${config.namespace}_` : "";

        switch (this.distanceStrategy) {
            case "cosine":
                idxDistanceFunction = "vector_cosine_ops";
                break;
            case "innerProduct":
                idxDistanceFunction = "vector_ip_ops";
                break;
            case "euclidean":
                idxDistanceFunction = "vector_l2_ops";
                break;
            default:
                throw new Error(`Unknown distance strategy: ${this.distanceStrategy}`);
        }

        const createIndexQuery = `CREATE INDEX IF NOT EXISTS ${prefix}${
            this.vectorColumnName
        }_embedding_hnsw_idx
        ON ${this.computedTableName} USING hnsw ((${
            this.vectorColumnName
        }::vector(${config.dimensions})) ${idxDistanceFunction})
        WITH (
            m=${config?.m || 16},
            ef_construction=${config?.efConstruction || 64}
        );`;

        try {
            await this.pgClient.query(createIndexQuery);
        } catch (e) {
            console.error(
                `Failed to create HNSW index on table ${this.computedTableName}, error: ${e}`
            );
        }
    }

    /**
     * Return documents selected using the maximal marginal relevance.
     * Maximal marginal relevance optimizes for similarity to the query AND
     * diversity among selected documents.
     * @param query Text to look up documents similar to.
     * @param options.k=4 Number of documents to return.
     * @param options.fetchK=20 Number of documents to fetch before passing to
     *     the MMR algorithm.
     * @param options.lambda=0.5 Number between 0 and 1 that determines the
     *     degree of diversity among the results, where 0 corresponds to maximum
     *     diversity and 1 to minimum diversity.
     * @returns List of documents selected by maximal marginal relevance.
     */
    async maxMarginalRelevanceSearch(
        query: string,
        options: MaxMarginalRelevanceSearchOptions<this["FilterType"]>
    ): Promise<Document[]> {
        const { k = 4, fetchK = 20, lambda = 0.5, filter } = options;
        const queryEmbedding = await this.embeddings.embedQuery(query);

        const docs = await this.searchPostgres(
            queryEmbedding,
            fetchK,
            filter,
            true
        );

        const embeddingList = docs.map((doc) =>
            JSON.parse(doc[0].metadata[this.vectorColumnName])
        );

        const mmrIndexes = maximalMarginalRelevance(
            queryEmbedding,
            embeddingList,
            lambda,
            k
        );

        const mmrDocs = mmrIndexes.map((index) => {
            const doc = docs[index][0];
            delete doc.metadata[this.vectorColumnName];
            return docs[index][0];
        });
        return mmrDocs;
    }
}