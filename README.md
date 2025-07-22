# Luminous Chatbot - Comprehensive Low Level Design (LLD)

## System Overview
This comprehensive LLD document provides detailed class diagrams, interfaces, protocols, and architectural patterns for the Luminous Chatbot system.

## 1. High-Level Architecture Overview

```mermaid
flowchart LR
    %% ================ Client Layer ================
    subgraph CL["Client Layer"]
        direction TB
        CLI["Client API Calls"]
        WEB["Web Interface"]
        MOB["Mobile App"]
    end

    %% ================ API Gateway ================
    subgraph AG["API Gateway & Auth"]
        direction TB
        MAIN["main.py\nFastAPI App"]
        AUTH["auth.py\nAPI Key Verification"]
        ENC["encryption.py\nRSA Encryption"]
        RATE["RateLimiter\nRate Limiting"]
        HEADER["HeaderValidator\nHeader Processing"]
        TOKEN["TokenAuthenticator\nToken Management"]
    end

    %% ================ Core Services ================
    subgraph CS["Core Services"]
        direction TB
        CHAT["chat_handler\nMain Chat Logic"]
        TASK["task_verification.py\nTask Validation"]
        LLM["llm_utils.py\nGemini Integration"]
        FEEDBACK["feedback_utils.py\nUser Feedback"]
        CONFIG["config.py\nConfiguration"]
        LOGGING["logging_setup.py\nLogging"]
    end

    %% ================ RAG System ================
    subgraph RS["RAG System"]
        direction TB
        RAGTOOL["RAGTool\nMain Controller"]
        SETTINGS["Settings\nConfig Manager"]
        METADATA["MetadataStore\nDocument Metadata"]
        DOCPROC["DocumentProcessor\nFile Processing"]
        EMBED["EmbeddingModel\nVector Generation"]
        VECTOR["PGVectorStore\nVector Storage"]
    end

    %% ================ Processing Pipeline ================
    subgraph PP["Document Processing"]
        direction TB
        EXTBASE["BaseExtractor\nAbstract Extractor"]
        PDFEXT["PDFExtractor"]
        DOCEXT["DocxExtractor"]
        CSVEXT["CSVExtractor"]
        IMGEXT["ImageExtractor"]
        TXTEXT["TextExtractor"]
        LLMEXT["LLMProcessedExtractor"]
    end

    %% ================ Advanced Processing ================
    subgraph AP["Advanced Processing"]
        direction TB
        FASTPDF["FastPDFProcessor"]
        DONUT["DonutPDFProcessor"]
        LLMPROC["LLMProcessor\nAI Analysis"]
        DEVICE["DeviceExtractor"]
        TAGS["TagDetector"]
    end

    %% ================ Caching Layer ================
    subgraph CC["Caching & Performance"]
        direction TB
        SEMANTIC["SemanticCache"]
        MEMORY["In-Memory Cache"]
        SIMILARITY["SimilaritySearch"]
    end

    %% ================ Database Layer ================
    subgraph DB["Database Layer"]
        direction TB
        DBHELP["db_helpers.py\nUtilities"]
        PGVECTOR["PostgreSQL + pgvector"]
        CONNPOOL["Connection Pool"]
        MIGRATION["Schema Migration"]
    end

    %% ================ Utilities ================
    subgraph UT["Data Models & Utilities"]
        direction TB
        MODELS["models.py\nPydantic Models"]
        TEXTUTIL["Text Utilities"]
        FILEUTIL["File Utilities"]
        SECLOG["Secure Logging"]
        SECDATA["Secure Data"]
    end

    %% ================ External Services ================
    subgraph ES["External Services"]
        direction TB
        GEMINI["Google Gemini API"]
        OPENAI["OpenAI API"]
        AUTOGEN["AutoGen Studio"]
        TESSERACT["Tesseract OCR"]
        POPPLER["Poppler PDF"]
    end

    %% ===== Main Flow =====
    CL --> AG
    AG --> CS
    CS --> RS
    RS --> PP
    PP --> AP
    
    %% ===== Core Connections =====
    CHAT --> TASK
    CHAT --> LLM
    CHAT --> FEEDBACK
    TASK --> SEMANTIC
    LLM --> GEMINI
    LLM --> OPENAI

    %% ===== RAG Connections =====
    RAGTOOL --> SETTINGS
    RAGTOOL --> METADATA
    RAGTOOL --> DOCPROC
    RAGTOOL --> EMBED
    RAGTOOL --> VECTOR
    RAGTOOL --> SEMANTIC
    EMBED --> VECTOR
    SEMANTIC --> MEMORY
    SEMANTIC --> SIMILARITY

    %% ===== Processing Connections =====
    DOCPROC --> EXTBASE
    EXTBASE --> PDFEXT
    EXTBASE --> DOCEXT
    EXTBASE --> CSVEXT
    EXTBASE --> IMGEXT
    EXTBASE --> TXTEXT
    EXTBASE --> LLMEXT
    PDFEXT --> FASTPDF
    PDFEXT --> DONUT
    PDFEXT --> POPPLER
    LLMEXT --> LLMPROC
    LLMPROC --> DEVICE
    LLMPROC --> TAGS
    LLMPROC --> AUTOGEN
    FASTPDF --> TESSERACT
    DONUT --> TESSERACT

    %% ===== Database Connections =====
    METADATA --> DBHELP
    VECTOR --> DBHELP
    DBHELP --> PGVECTOR
    DBHELP --> CONNPOOL
    DBHELP --> MIGRATION

    %% ===== Utilities Connections =====
    CHAT --> MODELS
    DOCPROC --> TEXTUTIL
    DOCPROC --> FILEUTIL
    LOGGING --> SECLOG
    ENC --> SECDATA

    %% ===== Styling =====
    classDef client fill:#D6EAF8,stroke:#1F618D,stroke-width:2px,rounded:4px
    classDef gateway fill:#D5F5E3,stroke:#27AE60,stroke-width:2px,rounded:4px
    classDef core fill:#FDEBD0,stroke:#F39C12,stroke-width:2px,rounded:4px
    classDef rag fill:#E8DAEF,stroke:#8E44AD,stroke-width:2px,rounded:4px
    classDef processing fill:#FADBD8,stroke:#C0392B,stroke-width:2px,rounded:4px
    classDef advanced fill:#D1F2EB,stroke:#148F77,stroke-width:2px,rounded:4px
    classDef cache fill:#FCF3CF,stroke:#B7950B,stroke-width:2px,rounded:4px
    classDef database fill:#D6DBDF,stroke:#34495E,stroke-width:2px,rounded:4px
    classDef utils fill:#EAECEE,stroke:#566573,stroke-width:2px,rounded:4px
    classDef external fill:#F9E79F,stroke:#B9770E,stroke-width:2px,rounded:4px

    class CL,CLI,WEB,MOB client
    class AG,MAIN,AUTH,ENC,RATE,HEADER,TOKEN gateway
    class CS,CHAT,TASK,LLM,FEEDBACK,CONFIG,LOGGING core
    class RS,RAGTOOL,SETTINGS,METADATA,DOCPROC,EMBED,VECTOR rag
    class PP,EXTBASE,PDFEXT,DOCEXT,CSVEXT,IMGEXT,TXTEXT,LLMEXT processing
    class AP,FASTPDF,DONUT,LLMPROC,DEVICE,TAGS advanced
    class CC,SEMANTIC,MEMORY,SIMILARITY cache
    class DB,DBHELP,PGVECTOR,CONNPOOL,MIGRATION database
    class UT,MODELS,TEXTUTIL,FILEUTIL,SECLOG,SECDATA utils
    class ES,GEMINI,OPENAI,AUTOGEN,TESSERACT,POPPLER external

    %% ===== Legend =====
    subgraph legend[" "]
        direction LR
        l1[Client]:::client
        l2[Gateway]:::gateway
        l3[Core]:::core
        l4[RAG]:::rag
        l5[Processing]:::processing
        l6[Advanced]:::advanced
        l7[Cache]:::cache
        l8[Database]:::database
        l9[Utils]:::utils
        l10[External]:::external
    end
```

## 2. Core Interfaces and Protocols

### 2.1 Authentication Protocol Interface

```mermaid
classDiagram
    class IAuthenticator {
        <<interface>>
        +verify_api_key(request: Request) bool
        +verify_admin_key(request: Request) bool
        +generate_token(user_data: Dict) str
        +validate_token(token: str) Tuple[bool, Dict]
    }

    class IEncryption {
        <<interface>>
        +encrypt(data: str) str
        +decrypt(encrypted_data: str) str
        +is_encrypted(data: str) bool
        +generate_auth_payload(user_id: str, auth_token: str, plant_id: str) str
        +parse_auth_payload(payload: str) Dict[str, str]
    }

    class IRateLimiter {
        <<interface>>
        +check_rate_limit(plant_id: str) bool
        +log_request(plant_id: str) None
        +enforce_rate_limit(plant_id: str) Tuple[bool, Optional[Dict]]
    }

    class AuthenticationHandler {
        -authenticator: IAuthenticator
        -encryption: IEncryption
        -rate_limiter: IRateLimiter
        +process_request(request: Request) AuthResult
        +validate_headers(headers: Dict) bool
    }

    AuthenticationHandler --> IAuthenticator
    AuthenticationHandler --> IEncryption
    AuthenticationHandler --> IRateLimiter
```

### 2.2 Document Processing Protocol

```mermaid
classDiagram
    class IDocumentExtractor {
        <<interface>>
        +can_extract(filepath: str) bool
        +extract(filepath: str) str
        +get_metadata(filepath: str) Dict[str, Any]
        +get_supported_formats() List[str]
    }

    class IDocumentProcessor {
        <<interface>>
        +process_files(directory: str, force_reprocess: bool) List[Document]
        +get_files_in_directory(directory: str) List[Tuple[str, str]]
        +chunk_document(text: str, metadata: Dict) List[Chunk]
    }

    class BaseExtractor {
        <<abstract>>
        +settings: Settings
        +can_extract(filepath: str)* bool
        +extract(filepath: str)* str
        +get_metadata(filepath: str) Dict[str, Any]
        #_clean_text(text: str) str
        #_extract_metadata(filepath: str) Dict
    }

    class PDFExtractor {
        +poppler_path: str
        +tesseract_cmd: str
        +can_extract(filepath: str) bool
        +extract(filepath: str) str
        -_extract_with_poppler(filepath: str) str
        -_extract_with_ocr(filepath: str) str
    }

    class DocxExtractor {
        +can_extract(filepath: str) bool
        +extract(filepath: str) str
        -_extract_paragraphs(doc: Document) str
        -_extract_tables(doc: Document) str
    }

    class LLMProcessedExtractor {
        +llm_client: LLMClient
        +can_extract(filepath: str) bool
        +extract(filepath: str) str
        -_enhance_with_llm(text: str) Dict[str, Any]
        -_extract_device_names(text: str) List[str]
    }

    IDocumentExtractor <|.. BaseExtractor
    BaseExtractor <|-- PDFExtractor
    BaseExtractor <|-- DocxExtractor
    BaseExtractor <|-- LLMProcessedExtractor
```

### 2.3 RAG System Protocol

```mermaid
classDiagram
    class IRAGTool {
        <<interface>>
        +retrieve(query: str, top_k: int) List[Dict]
        +process_files(directory: str) List[Document]
        +query(question: str, context: List[str]) str
        +sync_volume_and_embeddings() Dict[str, Any]
    }

    class IVectorStore {
        <<interface>>
        +store_documents(documents: List[Document]) bool
        +similarity_search(query: str, k: int) List[Document]
        +similarity_search_with_score(query: str, k: int) List[Tuple[Document, float]]
        +delete_documents(filter: Dict) None
    }

    class IMetadataStore {
        <<interface>>
        +save_metadata(filename: str, metadata: Dict) bool
        +load_metadata() Dict[str, Any]
        +delete_metadata(filename: str) bool
        +get_files_from_database(collection: str) Set[str]
    }

    class RAGTool {
        -settings: Settings
        -metadata_store: IMetadataStore
        -document_processor: IDocumentProcessor
        -embedding_model: IEmbeddingModel
        -vector_store: IVectorStore
        -metadata: Dict[str, Any]
        +retrieve(query: str, top_k: int) List[Dict]
        +process_files(directory: str) List[Document]
        +_async_init() None
        +_ensure_initialized_async() None
    }

    class PGVectorStore {
        -settings: Settings
        -embedding_model: IEmbeddingModel
        -store: PGVector
        +store_documents(documents: List[Document]) bool
        +retrieve(query: str, top_k: int, device_name: str, tags: List[str]) List[Dict]
        -_initialize_vector_store() bool
        -_apply_filters(query: str, device_name: str, tags: List[str]) str
    }

   class IEmbeddingModel {
        <<interface>>
        +embed_text(text: str) List[float]
        +embed_query(query: str) List[float]
        +get_dimension() int
    }

    class MetadataStore {
        -db_pool: asyncpg.Pool
        +save_metadata(filename: str, metadata: Dict) bool
        +load_metadata() Dict[str, Any]
        +delete_metadata(filename: str) bool
        -_ensure_metadata_tables() None
    }

    IRAGTool <|.. RAGTool
    IVectorStore <|.. PGVectorStore
    IMetadataStore <|.. MetadataStore
    RAGTool --> IMetadataStore
    RAGTool --> IVectorStore
    RAGTool --> IDocumentProcessor
    RAGTool --> IEmbeddingModel
```

## 3. Detailed Class Diagrams

### 3.1 API Gateway and Authentication Classes

```mermaid
classDiagram
    class FastAPIApp {
        -app: FastAPI
        -team_manager: TeamManager
        -semantic_cache: SemanticCache
        -rag_tool: RAGTool
        +startup() None
        +shutdown() None
        +chat_handler(request: Request) JSONResponse
        +upload_document(file: UploadFile) JSONResponse
        +list_documents() JSONResponse
        +health_check() JSONResponse
    }

    class EncryptionManager {
        -private_key: RSAPrivateKey
        -public_key: RSAPublicKey
        +encrypt(data: str) str
        +decrypt(encrypted_data: str) str
        +is_encrypted(data: str) bool
        +decrypt_if_encrypted(data: str) str
        +generate_auth_payload(user_id: str, auth_token: str, plant_id: str) str
        +parse_auth_payload(payload: str) Dict[str, str]
        +decrypt_headers(headers: Dict[str, Any]) Dict[str, Any]
    }

    class RateLimiter {
        -db_pool: asyncpg.Pool
        -requests_limit: int
        -window_hours: int
        +check_rate_limit(plant_id: str) bool
        +log_request(plant_id: str) None
        +enforce_rate_limit(plant_id: str) Tuple[bool, Optional[Dict]]
        -_cleanup_old_requests(plant_id: str) None
    }

    class HeaderValidator {
        +extract_headers(request: Request, decrypt_encrypted: bool) Dict[str, str]
        +validate_required_headers(headers: Dict[str, str], required: List[str]) Tuple[bool, str]
        -_decrypt_header_value(value: str) str
        -_parse_auth_payload(payload: str) Dict[str, str]
    }

    class TokenAuthenticator {
        -db_pool: asyncpg.Pool
        -cache_days: int
        +initialize_cache_table() None
        +validate_token(token: str) Tuple[bool, Optional[Dict]]
        +cache_token_result(token: str, is_valid: bool, token_data: Dict) None
        +cleanup_expired_tokens() None
        -_check_token_in_cache(token: str) Tuple[bool, Optional[Dict]]
    }

    FastAPIApp --> EncryptionManager
    FastAPIApp --> RateLimiter
    FastAPIApp --> HeaderValidator
    FastAPIApp --> TokenAuthenticator
```

### 3.2 Core Service Classes

```mermaid
classDiagram
    class ChatHandler {
        -rag_tool: RAGTool
        -semantic_cache: SemanticCache
        -team_manager: TeamManager
        +process_chat_request(request: ChatRequest) ChatResponse
        +verify_task(message: str) bool
        +generate_response(query: str, context: List[str]) str
        -_extract_user_context(request: Request) Dict[str, str]
        -_store_conversation(user_id: str, conversation_id: str, message: str, response: str) None
    }

    class TaskVerification {
        -llm_client: LLMClient
        -verification_prompt: str
        +verify_task(message: str) bool
        +is_luminous_related(message: str) bool
        -_call_verification_api(message: str) Dict[str, Any]
        -_parse_verification_result(result: Dict) bool
    }

    class LLMUtils {
        -gemini_client: GenerativeModel
        -openai_client: OpenAI
        -retry_config: RetryConfig
        +call_gemini_with_retry(prompt: str, max_retries: int) str
        +call_openai_with_retry(prompt: str, model: str) str
        +summarize_response(text: str) str
        -_handle_api_error(error: Exception) str
    }

    class FeedbackUtils {
        -db_pool: asyncpg.Pool
        +submit_feedback(payload: FeedbackPayload) bool
        +export_feedback_to_csv(filters: Dict) str
        +get_feedback_stats() Dict[str, Any]
        -_validate_feedback_data(payload: FeedbackPayload) bool
        -_store_feedback(payload: FeedbackPayload) None
    }

    class ConfigManager {
        +LOG_LEVEL: str
        +LOG_TYPE: str
        +DB_CONFIG: Dict[str, Any]
        +GEMINI_API_KEY: str
        +OPENAI_API_KEY: str
        +load_config() Dict[str, Any]
        +validate_config() bool
        +get_database_url() str
    }

    ChatHandler --> TaskVerification
    ChatHandler --> LLMUtils
    ChatHandler --> FeedbackUtils
    TaskVerification --> LLMUtils
```

### 3.3 Document Processing Classes

```mermaid
classDiagram
    class DocumentProcessor {
        -settings: Settings
        -metadata: Dict[str, Any]
        -extractors: List[IDocumentExtractor]
        +process_files(directory: str, force_reprocess: bool) List[Document]
        +get_files_in_directory_for_processing(directory: str) List[Tuple[str, str]]
        +chunk_document(text: str, metadata: Dict) List[Chunk]
        -_get_extractor_for_file(filepath: str) IDocumentExtractor
        -_process_single_file(filepath: str) Optional[Document]
    }

    class FastPDFProcessor {
        -tesseract_cmd: str
        -confidence_threshold: float
        +process_pdf(filepath: str) Dict[str, Any]
        +extract_text_blocks(page: fitz.Page) List[TextBlock]
        +detect_tables(page: fitz.Page) List[Table]
        +enhance_image_quality(image: Image) Image
        -_create_basic_blocks_from_ocr_data(ocr_data: Dict) List[TextBlock]
        -_detect_simple_tables(blocks: List[TextBlock]) List[Table]
    }

    class DonutPDFProcessor {
        -model: VisionEncoderDecoderModel
        -processor: DonutProcessor
        -device: str
        +process_pdf_page(image: Image) Dict[str, Any]
        +extract_structured_content(image: Image) Dict[str, Any]
        -_preprocess_image(image: Image) Image
        -_postprocess_output(output: Dict) Dict[str, Any]
    }

    class LLMProcessor {
        -llm_client: LLMClient
        -file_path: str
        +enhance_document_with_llm_tags(text: str, filename: str) Dict[str, Any]
        +extract_device_names(text: str) List[str]
        +generate_metadata_tags(content: str) List[str]
        +analyze_document_structure(text: str) Dict[str, Any]
        -_call_llm_for_analysis(text: str, prompt: str) str
        -_parse_llm_response(response: str) Dict[str, Any]
    }

    class DeviceExtractor {
        -device_patterns: List[Pattern]
        -luminous_models: Set[str]
        +extract_device_names_from_text(text: str) List[str]
        +validate_device_names(devices: List[str]) List[str]
        +get_device_specifications(device_name: str) Dict[str, Any]
        -_clean_device_names(devices: List[str]) List[str]
        -_match_known_devices(text: str) List[str]
    }

    DocumentProcessor --> FastPDFProcessor
    DocumentProcessor --> DonutPDFProcessor
    DocumentProcessor --> LLMProcessor
    LLMProcessor --> DeviceExtractor
```

### 3.4 Caching System Classes

```mermaid
classDiagram
    class SemanticCache {
        -db_pool: asyncpg.Pool
        -model: SentenceTransformer
        -similarity_threshold: float
        -cache_size: int
        -_cache: Dict[str, Any]
        -_cache_lock: asyncio.Lock
        +initialize() None
        +add_to_cache(query: str, response: Any, source: str, metadata: Dict) None
        +find_similar_query(query: str, threshold: float) Tuple[str, Any]
        +clear_memory_cache() None
        +get_cache_stats() Dict[str, Any]
        -_compute_similarity(query1: str, query2: str) float
        -_update_frequency(query: str) None
    }

    class InMemoryCache {
        -cache: Dict[str, CacheEntry]
        -max_size: int
        -ttl: int
        +get(key: str) Optional[Any]
        +set(key: str, value: Any, ttl: int) None
        +delete(key: str) bool
        +clear() None
        +size() int
        -_evict_expired() None
        -_evict_lru() None
    }

    class SimilaritySearch {
        -embedding_model: IEmbeddingModel
        -vector_store: IVectorStore
        +search_similar_documents(query: str, top_k: int) List[Document]
        +search_with_filters(query: str, filters: Dict, top_k: int) List[Document]
        +compute_similarity_score(text1: str, text2: str) float
        -_prepare_query_embedding(query: str) List[float]
        -_apply_semantic_filters(results: List[Document], filters: Dict) List[Document]
    }

    class CacheEntry {
        +key: str
        +value: Any
        +created_at: datetime
        +last_accessed: datetime
        +access_count: int
        +ttl: int
        +is_expired() bool
        +update_access() None
    }

    SemanticCache --> InMemoryCache
    SimilaritySearch --> IEmbeddingModel
    SimilaritySearch --> IVectorStore
    InMemoryCache --> CacheEntry
```

### 3.5 Database Layer Classes

```mermaid
classDiagram
    class DatabaseHelper {
        -db_pool: asyncpg.Pool
        +initialize_db_pool() asyncpg.Pool
        +get_db_connection() asyncpg.Connection
        +close_db_pool() None
        +execute_query(query: str, params: List) Any
        +fetch_all(query: str, params: List) List[Record]
        +add_to_context(user_id: str, conversation_id: str, message: str, response: str) None
        -_create_connection_pool() asyncpg.Pool
        -_validate_connection(conn: asyncpg.Connection) bool
    }

    class ConnectionPoolManager {
        -pools: Dict[str, asyncpg.Pool]
        -max_connections: int
        -min_connections: int
        +get_pool(database: str) asyncpg.Pool
        +create_pool(database: str, config: Dict) asyncpg.Pool
        +close_pool(database: str) None
        +health_check() Dict[str, bool]
        -_monitor_connections() None
        -_cleanup_idle_connections() None
    }

    class SchemaMigration {
        -db_pool: asyncpg.Pool
        -migration_scripts: List[str]
        +run_migrations() bool
        +create_tables() None
        +migrate_schema(version: str) bool
        +rollback_migration(version: str) bool
        +get_current_version() str
        -_execute_migration_script(script: str) bool
        -_update_migration_history(version: str) None
    }

    class PostgreSQLAdapter {
        -connection_string: str
        -pool: asyncpg.Pool
        +connect() None
        +disconnect() None
        +execute(query: str, params: List) Any
        +fetch_one(query: str, params: List) Optional[Record]
        +fetch_many(query: str, params: List) List[Record]
        +begin_transaction() asyncpg.Transaction
        +commit_transaction(tx: asyncpg.Transaction) None
        +rollback_transaction(tx: asyncpg.Transaction) None
    }

    DatabaseHelper --> ConnectionPoolManager
    DatabaseHelper --> SchemaMigration
    DatabaseHelper --> PostgreSQLAdapter
```

## 4. Data Models and DTOs

### 4.1 Request/Response Models

```mermaid
classDiagram
    class ChatRequest {
        +conversation_id: str
        +user_id: str
        +message: str
        +context: Optional[Dict[str, Any]]
        +timestamp: datetime
    }

    class ChatResponse {
        +conversation_id: str
        +response: Dict[str, Any]
        +status: bool
        +timestamp: datetime
        +metadata: Optional[Dict[str, Any]]
    }

    class DocumentUploadRequest {
        +file: UploadFile
        +directory: str
        +metadata: Optional[Dict[str, Any]]
        +force_reprocess: bool
    }

    class DocumentUploadResponse {
        +status: str
        +message: str
        +chunks_created: int
        +filepath: str
        +file_type: str
        +llm_detected_tags: Optional[List[str]]
    }

    class FeedbackPayload {
        +conversation_id: str
        +rating: int
        +feedback: Optional[str]
        +inverter_id: Optional[str]
        +inverter_name: Optional[str]
        +embedded_client: Optional[str]
        +user_name: Optional[str]
        +user_email: Optional[str]
        +user_phone: Optional[str]
    }

    class CacheEntry {
        +query: str
        +response: Dict[str, Any]
        +source: str
        +metadata: Optional[Dict[str, Any]]
    }

    class CacheUpdateRequest {
        +new_response: Dict[str, Any]
        +metadata: Optional[Dict[str, Any]]
    }
```

### 4.2 Internal Data Models

```mermaid
classDiagram
    class Document {
        +filename: str
        +filepath: str
        +content: str
        +metadata: Dict[str, Any]
        +chunks: List[Chunk]
        +created_at: datetime
        +last_modified: datetime
    }

    class Chunk {
        +id: str
        +document_id: str
        +content: str
        +metadata: Dict[str, Any]
        +embedding: Optional[List[float]]
        +chunk_index: int
        +chunk_type: str
    }

    class DeviceInfo {
        +device_name: str
        +model_number: str
        +specifications: Dict[str, Any]
        +manual_references: List[str]
        +supported_features: List[str]
    }

    class ProcessingResult {
        +success: bool
        +message: str
        +data: Optional[Dict[str, Any]]
        +error: Optional[str]
        +processing_time: float
    }

    class AuthResult {
        +authenticated: bool
        +user_id: Optional[str]
        +plant_id: Optional[str]
        +permissions: List[str]
        +rate_limit_remaining: int
    }

    Document --> Chunk
    ProcessingResult --> Document
```

## 5. Communication Protocols

### 5.1 HTTP API Protocol

```mermaid
sequenceDiagram
    participant C as Client
    participant AG as API Gateway
    participant A as Auth Service
    participant CS as Core Service
    participant RS as RAG Service
    participant DB as Database

    C->>AG: POST /api/chat
    AG->>A: Validate API Key
    A-->>AG: Auth Result
    AG->>A: Check Rate Limit
    A-->>AG: Rate Limit Status
    AG->>CS: Process Chat Request
    CS->>RS: Query RAG System
    RS->>DB: Vector Search
    DB-->>RS: Relevant Documents
    RS-->>CS: Context + Documents
    CS->>CS: Generate Response
    CS-->>AG: Chat Response
    AG-->>C: JSON Response
```

### 5.2 Document Processing Protocol

```mermaid
sequenceDiagram
    participant C as Client
    participant AG as API Gateway
    participant DP as Document Processor
    participant E as Extractor
    participant LLM as LLM Processor
    participant VS as Vector Store
    participant MS as Metadata Store

    C->>AG: POST /api/documents/upload
    AG->>DP: Process Document
    DP->>E: Extract Content
    E-->>DP: Raw Text
    DP->>LLM: Enhance with AI
    LLM-->>DP: Enhanced Content + Metadata
    DP->>DP: Create Chunks
    DP->>VS: Store Embeddings
    DP->>MS: Store Metadata
    VS-->>DP: Storage Confirmation
    MS-->>DP: Metadata Saved
    DP-->>AG: Processing Complete
    AG-->>C: Upload Response
```

## 6. Error Handling and Exception Hierarchy

```mermaid
classDiagram
    class BaseError {
        <<abstract>>
        +message: str
        +error_code: str
        +timestamp: datetime
        +context: Dict[str, Any]
    }

    class AuthenticationError {
        +invalid_credentials: bool
        +rate_limit_exceeded: bool
    }

    class DocumentProcessingError {
        +file_path: str
        +processing_stage: str
        +extractor_type: str
    }

    class RAGSystemError {
        +component: str
        +operation: str
        +query: Optional[str]
    }

    class DatabaseError {
        +connection_lost: bool
        +query: str
        +table: str
    }

    class ExternalServiceError {
        +service_name: str
        +api_endpoint: str
        +http_status: int
    }

    BaseError <|-- AuthenticationError
    BaseError <|-- DocumentProcessingError
    BaseError <|-- RAGSystemError
    BaseError <|-- DatabaseError
    BaseError <|-- ExternalServiceError
```

## 7. Performance and Scalability Considerations

### 7.1 Async Processing Architecture

```mermaid
flowchart TD
    subgraph "Async Processing Pipeline"
        A[Request Received] --> B{Authentication Required?}
        B -->|Yes| C[Async Auth Validation]
        B -->|No| D[Rate Limit Check]
        C --> D
        D --> E{Cache Hit?}
        E -->|Yes| F[Return Cached Response]
        E -->|No| G[Async RAG Processing]
        G --> H[Parallel Document Search]
        G --> I[Parallel Embedding Generation]
        H --> J[Merge Results]
        I --> J
        J --> K[LLM Response Generation]
        K --> L[Update Cache Async]
        K --> M[Return Response]
        L --> N[Background Cleanup]
    end
```

### 7.2 Caching Strategy

```mermaid
flowchart LR
    subgraph "Multi-Layer Caching"
        A[Request] --> B[L1: In-Memory Cache]
        B -->|Miss| C[L2: Redis Cache]
        C -->|Miss| D[L3: Database Cache]
        D -->|Miss| E[Generate Response]
        E --> F[Store in All Layers]
        
        G[Cache Invalidation]
        G --> H[Document Update]
        G --> I[Time-based Expiry]
        G --> J[LRU Eviction]
    end
```

## 8. Security Architecture

### 8.1 Security Layers

```mermaid
flowchart TB
    subgraph "Security Architecture"
        A[Client Request] --> B[TLS/SSL Termination]
        B --> C[API Gateway Security]
        C --> D[Rate Limiting]
        D --> E[API Key Validation]
        E --> F[RSA Encryption/Decryption]
        F --> G[Header Validation]
        G --> H[Token Authentication]
        H --> I[Authorization Check]
        I --> J[Secure Data Processing]
        J --> K[Audit Logging]
        K --> L[Response Encryption]
        L --> M[Secure Response]
    end
```

### 8.2 Data Protection Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant E as Encryption Layer
    participant A as Auth Service
    participant P as Processing Layer
    participant DB as Database
    participant L as Audit Log

    C->>E: Encrypted Request
    E->>E: Decrypt Headers
    E->>A: Validate Credentials
    A->>L: Log Auth Attempt
    A-->>E: Auth Result
    E->>P: Process with Secure Context
    P->>DB: Encrypted Query
    DB-->>P: Encrypted Response
    P->>L: Log Processing
    P-->>E: Secure Result
    E->>E: Encrypt Response
    E-->>C: Encrypted Response
```

## 9. Monitoring and Observability

### 9.1 Monitoring Architecture

```mermaid
flowchart TD
    subgraph "Monitoring Stack"
        A[Application Metrics] --> D[Prometheus]
        B[System Metrics] --> D
        C[Custom Metrics] --> D
        D --> E[Grafana Dashboard]
        
        F[Application Logs] --> G[Structured Logging]
        G --> H[Log Aggregation]
        H --> I[Kibana/LogViewer]
        
        J[Traces] --> K[Jaeger/Zipkin]
        K --> L[Distributed Tracing]
        
        M[Health Checks] --> N[Service Discovery]
        N --> O[Load Balancer]
    end
```

### 9.2 Key Metrics to Monitor

```mermaid
classDiagram
    class SystemMetrics {
        +cpu_usage: float
        +memory_usage: float
        +disk_io: float
        +network_io: float
        +connection_pool_size: int
    }

    class ApplicationMetrics {
        +request_rate: float
        +response_time: float
        +error_rate: float
        +cache_hit_ratio: float
        +document_processing_time: float
    }

    class BusinessMetrics {
        +active_users: int
        +documents_processed: int
        +queries_per_minute: int
        +feedback_score: float
        +device_queries: int
    }

    class SecurityMetrics {
        +failed_auth_attempts: int
        +rate_limit_violations: int
        +suspicious_activities: int
        +data_access_logs: int
    }
```

## 10. Deployment Architecture

### 10.1 Container Architecture

```mermaid
flowchart TD
    subgraph "Kubernetes Deployment"
        subgraph "API Tier"
            A[FastAPI Pod 1]
            B[FastAPI Pod 2]
            C[FastAPI Pod 3]
        end
        
        subgraph "Processing Tier"
            D[RAG Service Pod 1]
            E[RAG Service Pod 2]
            F[Background Worker Pod]
        end
        
        subgraph "Data Tier"
            G[PostgreSQL Primary]
            H[PostgreSQL Replica]
            I[Redis Cache]
        end
        
        subgraph "External Services"
            J[Gemini API]
            K[OpenAI API]
            L[Object Storage]
        end
        
        M[Load Balancer] --> A
        M --> B
        M --> C
        
        A --> D
        B --> E
        C --> F
        
        D --> G
        E --> H
        F --> I
        
        D --> J
        E --> K
        F --> L
    end
```

### 10.2 CI/CD Pipeline

```mermaid
flowchart LR
    A[Code Commit] --> B[Unit Tests]
    B --> C[Integration Tests]
    C --> D[Security Scan]
    D --> E[Build Container]
    E --> F[Push to Registry]
    F --> G[Deploy to Staging]
    G --> H[E2E Tests]
    H --> I[Performance Tests]
    I --> J[Deploy to Production]
    J --> K[Health Check]
    K --> L[Monitoring Alert]
```

## 11. Configuration Management

### 11.1 Configuration Classes

```mermaid
classDiagram
    class BaseConfig {
        <<abstract>>
        +environment: str
        +debug: bool
        +log_level: str
        +validate() bool
    }

    class DatabaseConfig {
        +host: str
        +port: int
        +username: str
        +password: SecureString
        +database: str
        +pool_size: int
        +connection_timeout: int
    }

    class APIConfig {
        +host: str
        +port: int
        +workers: int
        +timeout: int
        +max_request_size: int
        +cors_origins: List[str]
    }

    class SecurityConfig {
        +api_key: SecureString
        +admin_key: SecureString
        +private_key: SecureString
        +public_key: SecureString
        +token_expiry: int
        +rate_limit: int
    }

    class ExternalServiceConfig {
        +gemini_api_key: SecureString
        +openai_api_key: SecureString
        +tesseract_path: str
        +poppler_path: str
        +timeout: int
    }

    BaseConfig <|-- DatabaseConfig
    BaseConfig <|-- APIConfig
    BaseConfig <|-- SecurityConfig
    BaseConfig <|-- ExternalServiceConfig
```

## 12. Testing Strategy

### 12.1 Test Architecture

```mermaid
flowchart TD
    subgraph "Testing Pyramid"
        A[Unit Tests] --> B[Integration Tests]
        B --> C[Contract Tests]
        C --> D[End-to-End Tests]
        D --> E[Performance Tests]
        E --> F[Security Tests]
    end
    
    subgraph "Test Types"
        G[Component Tests]
        H[API Tests]
        I[Database Tests]
        J[External Service Tests]
        K[Load Tests]
        L[Penetration Tests]
    end
```

### 12.2 Test Data Management

```mermaid
classDiagram
    class TestDataManager {
        +create_test_documents() List[Document]
        +create_test_users() List[User]
        +setup_test_database() None
        +cleanup_test_data() None
        +generate_sample_queries() List[str]
    }

    class MockExternalServices {
        +mock_gemini_api() MockGeminiAPI
        +mock_openai_api() MockOpenAIAPI
        +mock_file_storage() MockFileStorage
    }

    class TestFixtures {
        +pdf_samples: List[str]
        +docx_samples: List[str]
        +image_samples: List[str]
        +device_data: Dict[str, Any]
        +chat_scenarios: List[ChatScenario]
    }
```

## Summary

This comprehensive LLD provides:

1. **Complete System Architecture** - High-level component relationships
2. **Detailed Interface Definitions** - Clear contracts between components
3. **Class Diagrams** - Internal structure of each major component
4. **Communication Protocols** - How components interact
5. **Data Models** - Request/response and internal data structures
6. **Security Architecture** - Multi-layer security approach
7. **Performance Considerations** - Async processing and caching strategies
8. **Monitoring & Observability** - Comprehensive monitoring approach
9. **Deployment Architecture** - Container-based deployment strategy
10. **Configuration Management** - Structured configuration approach
11. **Testing Strategy** - Comprehensive testing pyramid

This LLD serves as a complete technical blueprint for the Luminous Chatbot system, providing developers with detailed guidance for implementation, maintenance, and scaling.

## Key Component Details

### 1. Authentication & Security Components
- **main.py**: FastAPI application entry point with 30+ endpoints
- **auth.py**: API key verification using SHA256 hashing
- **encryption.py**: RSA encryption for sensitive data (headers, payloads)
- **RateLimiter**: Plant-based rate limiting with sliding window
- **HeaderValidator**: Request header extraction and validation
- **TokenAuthenticator**: Token caching and validation system

### 2. Core Processing Functions
- **chat_handler**: Main chat processing logic with error handling
- **task_verification**: Task validation using Gemini API
- **llm_utils**: Gemini API integration with retry mechanism
- **feedback_utils**: User feedback collection and export
- **SemanticCache**: Intelligent query caching using sentence transformers

### 3. RAG System Components
- **RAGTool**: Main coordinator for retrieval-augmented generation
- **DocumentProcessor**: File processing pipeline manager
- **EmbeddingModel**: Vector embedding generation
- **PGVectorStore**: PostgreSQL + pgvector integration
- **MetadataStore**: Document metadata management

### 4. Document Processing Pipeline
- **BaseExtractor**: Abstract base for all extractors
- **PDFExtractor**: PDF text extraction with OCR fallback
- **DocxExtractor**: Word document processing
- **CSVExtractor**: CSV file processing with device detection
- **ImageExtractor**: OCR for image files
- **LLMProcessedExtractor**: AI-enhanced content extraction

### 5. Advanced Processing Features
- **FastPDFProcessor**: High-performance PDF processing
- **DonutPDFProcessor**: Advanced OCR using Donut model
- **LLMProcessor**: AI-powered content analysis
- **DeviceExtractor**: Luminous device name detection
- **TagDetector**: Automated metadata tag generation

### 6. Database & Utilities
- **db_helpers**: Database connection management
- **PostgreSQL + pgvector**: Vector database for embeddings
- **Secure logging**: PII-safe logging framework
- **Configuration management**: Environment-based settings

## Data Flow Summary

1. **Request Flow**: Client → Authentication → Rate Limiting → Main Handler
2. **Document Processing**: Upload → Extraction → Embedding → Vector Storage
3. **Query Processing**: Query → Semantic Cache Check → Vector Search → LLM Processing → Response
4. **Caching Strategy**: In-memory + Database semantic caching for performance
5. **Security**: End-to-end encryption with secure data handling

## Key Integration Points
- **External APIs**: Gemini, OpenAI, AutoGen Studio
- **OCR Tools**: Tesseract, Poppler, Donut model
- **Vector Database**: PostgreSQL with pgvector extension
- **Caching**: Multi-layer caching (memory + database)
- **Authentication**: Multi-tier security with encryption
