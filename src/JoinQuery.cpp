#include "JoinQuery.hpp"
#include <assert.h>
#include <fstream>
#include <thread>


// Aggiunti
#include <atomic>
#include <vector>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>


using namespace std;

// Struttura per lineitem (l_orderkey, l_quantity)
struct LineItemRecord {
   int orderkey;
   double quantity;
};

// Struttura per orders (o_orderkey, o_custkey)
struct OrderRecord {
   int orderkey;
   int custkey;
};

// Struttura per customer (c_custkey, c_mktsegment)
struct CustomerRecord {
   int custkey;
   char mktsegment[11];
};

// Contenitori per i record relativi alle diverse tabelle
vector<LineItemRecord> lineitemRecords;
vector<OrderRecord> ordersRecords;
vector<CustomerRecord> customerRecords;

class MemoryMappedFile
{
   int handle = -1;           // hanlde del file
   uintptr_t size = 0;        // dimensione della mappatura
   char* mapping = nullptr;   // puntatore alla mappatura in memoria

public:
   // Distruttore: chiude la mappatura se aperta
   ~MemoryMappedFile() { close(); }

   // Definisce il metodo per aprire un file e mappalo in memoria
   bool open(const char* file);

   // Definisce il metodo per chiudere il file e rilasciare il mapping
   void close();

   // Fornisce i puntatore all'inizio e alla fine della mappatura
   const char* begin() const { return mapping; }
   const char* end() const { return mapping+size; }
};


bool MemoryMappedFile::open(const char* file)
{
   // Chiude eventuali file aperti precedentemente
   close();

   // Apre il file in sola lettura
   // Ritornando il file descriptor, che è un intero non negativo se l'apertura ha successo
   int h = ::open(file, O_RDONLY);

   // Se h<0, l'apertura del file è fallita
   if (h<0) 
      return false;

   // Sposta il cursore alla fine
   lseek(h,0,SEEK_END);

   // Ottiene la posizione corrente del cursore
   // Essendo alla fine del file, lseek torna la dimensione del file in byte (char)
   size=lseek(h,0,SEEK_CUR);

   // Crea il mapping in memoria del file mediante la chiamata di sistema mmap
   // Si passano dei parametri per specificare:
   // - l'indirizzo suggerito (nullptr = sistema decide)
   // - la dimensione del mapping
   // - la protezione (sola lettura)
   // - il tipo di mapping (condiviso)
   // - il file descriptor
   // - l'offset nel file (0 = dall'inizio)   
   auto m = mmap(nullptr,        // Indirizzo suggerito (nullptr = sistema decide)
               size,             // Dimensione mapping
               PROT_READ,        // Protezione (sola lettura)
               MAP_SHARED,       // Tipo di mapping (condiviso)
               h,                // File descriptor
               0);               // Offset nel file
   
   // Se il mapping fallisce, mmap ritorna MAP_FAILED
   if (m == MAP_FAILED) {
      // Chiude il file descriptor aperto ed esce dal metodo con false
      ::close(h);
      return false;
   }

   // Salva il file descriptor nella variabile di classe
   handle = h;                     

   // Converte il puntatore void* in char* e lo salva nella variabile di classe
   mapping = static_cast<char*>(m); 
   
   return true;
}

void MemoryMappedFile::close()
{
   // Se il file è aperto, rilascia la mappatura e chiudi il file
   if (handle>=0) {
      munmap(mapping, size);
      ::close(handle);
      handle=-1;
   }
}


// SWAR (SIMD Within A Register) implementations
// for finding characters in the input stream
// processes 8 bytes at a time using 64-bit integers
// and bitwise operations
// Used to avoid SIMD instruction set dependencies
// and improve portability across different CPUs
// Find character 'c' in the range [iter, limit)
template <char c>
static inline const char* find(const char* iter, const char* limit)
{
   // Precompute limit for 8-byte blocks
   auto limit8=limit-8;

   // Process blocks of 8 bytes at a time
   // until reaching the last 8-byte block
   while (iter<limit8) {

      // Load a block of 8 bytes at a time
      auto block = *reinterpret_cast<const uint64_t*>(iter);

      // Define the pattern to search for
      constexpr uint64_t pattern = (static_cast<uint64_t>(c)<<56)|(static_cast<uint64_t>(c)<<48)|(static_cast<uint64_t>(c)<<40)|(static_cast<uint64_t>(c)<<32)|(static_cast<uint64_t>(c)<<24)|(static_cast<uint64_t>(c)<<16)|(static_cast<uint64_t>(c)<<8)|(static_cast<uint64_t>(c)<<0);
      
      // Use SWAR technique to find matching bytes
      constexpr uint64_t lowerBits = 0x7F7F7F7F7F7F7F7Full;
      constexpr uint64_t topBits = ~lowerBits;
      uint64_t asciiChars = (~block)&topBits;
      uint64_t foundPattern = ~((((block&lowerBits) ^ pattern)+lowerBits)&topBits);
      uint64_t matches = foundPattern & asciiChars;

      // If matches found, return the position of the first match
      if (matches) {
         return iter+__builtin_ctzll(matches)/8;
      } else {
         iter+=8;
      }
   }

   // Process remaining bytes one by one
   while ((iter!=limit)&&((*iter)!=c)) 
      ++iter;
   
   // Return pointer to the found character or limit if not found
   return iter;
}

// Find the n-th occurrence of character 'c' in the range [iter, limit)
// Returns pointer to the n-th occurrence or limit if not found
template <char c>
static inline const char* findNth(const char* iter, const char* limit, unsigned n)
{
   // Precompute limit for 8-byte blocks
   auto limit8=limit-8;

   // Process blocks of 8 bytes at a time
   // until reaching the last 8-byte block
   while (iter<limit8) {
      // Load a block of 8 bytes at a time
      auto block = *reinterpret_cast<const uint64_t*>(iter);

      // Define the pattern to search for
      constexpr uint64_t pattern = (static_cast<uint64_t>(c)<<56)|(static_cast<uint64_t>(c)<<48)|(static_cast<uint64_t>(c)<<40)|(static_cast<uint64_t>(c)<<32)|(static_cast<uint64_t>(c)<<24)|(static_cast<uint64_t>(c)<<16)|(static_cast<uint64_t>(c)<<8)|(static_cast<uint64_t>(c)<<0);

      // Use SWAR technique to find matching bytes
      constexpr uint64_t lowerBits = 0x7F7F7F7F7F7F7F7Full;
      constexpr uint64_t topBits = ~lowerBits;
      uint64_t asciiChars = (~block)&topBits;
      uint64_t foundPattern = ~((((block&lowerBits) ^ pattern)+lowerBits)&topBits);
      uint64_t matches = foundPattern & asciiChars;

      // If matches found, check if we have enough occurrences
      if (matches) {
         // Count number of matches in this block
         unsigned hits=__builtin_popcountll(matches);

         // If number of matches is less than n
         if (hits<n) {
            // Decrease n
            n-=hits;
            // Advance iterator by 8 bytes (move to next block)
            iter+=8;
         } else {
            // Have found the n-th occurrence in this block
            while (n>1) {
               // In case of multiple matches, update matches to remove the least significant set bit
               matches&=matches-1;
               // Decrease n
               --n;
            }
            // Now matches contains only the n-th occurrence
            // Return pointer to the n-th occurrence
            return iter+__builtin_ctzll(matches)/8;
         }
      } else {
         // No matches in this block, advance iterator by 8 bytes (next block)
         iter+=8;
      }
   }

   // Process remaining bytes one by one
   while ((iter!=limit)&&((*iter)!=c)) 
      ++iter;

   return iter;
}


// Get the chunk boundary for dividing work among threads
// Ensures that chunks start and end at line boundaries
// to avoid splitting lines between threads
// chunk: index of the chunk (0-based)
// chunkCount: total number of chunks
// Returns pointer to the start of the chunk
static const char* getChunkBoundary(const char* begin, const char* end, unsigned chunk, unsigned chunkCount)
{
   // Chunk 0 starts at begin
   if (chunk==0) 
      return begin;
   
   // Last chunk ends at end
   if (chunk==chunkCount) 
      return end;

   // sep è uguale al primo chuck di file (ad es. con chunkCount=4, chunk=1 => 25% del file)
   auto sep=begin+((end-begin)*chunk/chunkCount);

   // Sistemo sep affinchè termini sempre con la fine di una riga 
   sep = find<'\n'>(sep, end);

   // Se non sono arrivato alla fine del file
   // avanzo di un carattere per posizionarmi alla prima riga del chunk successivo
   if (sep != end) 
      sep++;
   
   // Ritorna il puntatore al boundary del chunk successivo
   return sep;
}

// Funzione helper ottimizzata per il parsing di interi
static inline int parseInteger(const char*& p, const char* limit, char delimiter)
{
   int value = 0;
   while (p != limit && *p != delimiter) {
      value = value * 10 + (*p - '0');
      ++p;
   }
   return value;
}

// Funzione helper ottimizzata per il parsing di double
static inline double parseDouble(const char*& p, const char* limit, char delimiter)
{
   double value = 0.0;
   while (p != limit && *p != delimiter && *p != '.') {
      value = value * 10.0 + (*p - '0');
      ++p;
   }
   if (p != limit && *p == '.') {
      ++p;
      double base = 0.1;
      while (p != limit && *p != delimiter) {
         value += base * (*p - '0');
         base *= 0.1;
         ++p;
      }
   }
   return value;
}

// Funzione helper ottimizzata per il parsing di stringhe
static inline void parseString(const char*& p, const char* limit, char* output, int maxLen, char delimiter)
{
   int idx = 0;
   while (p != limit && *p != delimiter && idx < maxLen - 1) {
      output[idx++] = *p;
      ++p;
   }
   output[idx] = '\0';
}

// Versione ottimizzata: estrae lineitem con reserve pre-allocato
static size_t processLineItem(const char* filename, vector<LineItemRecord>& allRecords)
{
   // Carico il file in memoria
   MemoryMappedFile in;
   in.open(filename);

   // Creo i thread per il processamento in parallelo
   vector<thread> threads;
   const unsigned chunkCount = thread::hardware_concurrency();
   // Numero di worker (thread) dopo avere verificato se chunkCount è valido o meno
   const unsigned workers = chunkCount ? chunkCount : 1;

   // container per-thread per-record storage con pre-allocazione
   vector<vector<LineItemRecord>> recordsPerThread(workers);
   for (auto& v : recordsPerThread) {
      v.reserve(1500000);  // Pre-allocazione per evitare riallocazioni
   }

   // Launch threads to process each chunk
   for (unsigned index = 0; index != workers; ++index) {
      threads.push_back(thread([index, &in, workers, &recordsPerThread]() 
      {
         // Ottiene i boundary del chunk per il singolo thread
         auto from = getChunkBoundary(in.begin(), in.end(), index, workers);
         auto to = getChunkBoundary(in.begin(), in.end(), index+1, workers);

         // Riferimento al vettore locale per i record di questo thread
         auto& localRecords = recordsPerThread[index];
         
         // Processa ogni riga nel chunk
         for (auto iter = from; iter != to;) {
            // leggi primo campo (l_orderkey)
            const char* p = iter;
            int orderkey = parseInteger(p, to, '|');
            ++p;  // skip '|'

            // trova l'inizio del quinto campo (dopo la 4ª '|')
            auto field5 = findNth<'|'>(iter, to, 4);
            if (field5 != to) ++field5;

            // parse del campo quantity come double
            const char* q = field5;
            double qty = parseDouble(q, to, '|');

            // salva il record relativo ad una riga
            localRecords.push_back(LineItemRecord{orderkey, qty});

            // avanza alla fine della riga
            iter = find<'\n'>(iter, to);
            if (iter != to) 
               ++iter;
         }
         
      }));
   }

   // wait for all threads
   for (auto& t : threads) 
      t.join();

   // aggrega i record da tutti i thread
   size_t totalRecords = 0;
   for (const auto& v : recordsPerThread) {
      totalRecords += v.size();
   }
   
   // Usa reserve per allocare lo spazio necessario in una sola volta pari al numero totale di record
   allRecords.reserve(totalRecords);
   for (auto& v : recordsPerThread) {
      allRecords.insert(allRecords.end(), 
                       make_move_iterator(v.begin()), 
                       make_move_iterator(v.end()));
   }

   // Chiude il file mappato
   in.close();

   return totalRecords;
}

// Versione ottimizzata: estrae orders con reserve pre-allocato
static size_t processOrders(const char* filename, vector<OrderRecord>& allRecords)
{
   MemoryMappedFile in;
   in.open(filename);

   vector<thread> threads;
   const unsigned chunkCount = thread::hardware_concurrency();
   const unsigned workers = chunkCount ? chunkCount : 1;

   // container per-thread per-record storage con pre-allocazione
   vector<vector<OrderRecord>> recordsPerThread(workers);
   for (auto& v : recordsPerThread) {
      v.reserve(1500000);
   }

   // Launch threads to process each chunk
   for (unsigned index = 0; index != workers; ++index) {
      threads.push_back(thread([index, &in, workers, &recordsPerThread]() {
         auto from = getChunkBoundary(in.begin(), in.end(), index, workers);
         auto to = getChunkBoundary(in.begin(), in.end(), index+1, workers);

         auto& localRecords = recordsPerThread[index];
         
         // Processa ogni riga nel chunk
         for (auto iter = from; iter != to;) {
            // leggi primo campo (o_orderkey)
            const char* p = iter;
            int orderkey = parseInteger(p, to, '|');
            ++p;  // skip '|'

            // leggi secondo campo (o_custkey)
            int custkey = parseInteger(p, to, '|');

            // salva il record
            localRecords.push_back(OrderRecord{orderkey, custkey});

            // avanza alla fine della riga
            iter = find<'\n'>(iter, to);
            if (iter != to) ++iter;
         }
         
      }));
   }

   // sync for all threads
   for (auto& t : threads) 
      t.join();

   // aggrega i record da tutti i thread
   size_t totalRecords = 0;
   for (const auto& v : recordsPerThread) {
      totalRecords += v.size();
   }
   
   allRecords.reserve(totalRecords);
   for (auto& v : recordsPerThread) {
      allRecords.insert(allRecords.end(), 
                       make_move_iterator(v.begin()), 
                       make_move_iterator(v.end()));
   }

   in.close();
   return totalRecords;
}

// Versione ottimizzata: estrae customer con reserve pre-allocato
static size_t processCustomer(const char* filename, vector<CustomerRecord>& allRecords)
{
   MemoryMappedFile in;
   in.open(filename);

   vector<thread> threads;
   const unsigned chunkCount = thread::hardware_concurrency();
   const unsigned workers = chunkCount ? chunkCount : 1;

   // container per-thread per-record storage con pre-allocazione
   vector<vector<CustomerRecord>> recordsPerThread(workers);
   for (auto& v : recordsPerThread) {
      v.reserve(150000);
   }

   // Launch threads to process each chunk
   for (unsigned index = 0; index != workers; ++index) {
      threads.push_back(thread([index, &in, workers, &recordsPerThread]() {
         auto from = getChunkBoundary(in.begin(), in.end(), index, workers);
         auto to = getChunkBoundary(in.begin(), in.end(), index+1, workers);

         auto& localRecords = recordsPerThread[index];
         
         // Processa ogni riga nel chunk
         for (auto iter = from; iter != to;) {
            // leggi primo campo (c_custkey)
            const char* p = iter;
            int custkey = parseInteger(p, to, '|');
            ++p;  // skip '|'

            // trova l'inizio del settimo campo (dopo la 6ª '|')
            auto field7 = findNth<'|'>(iter, to, 6);
            if (field7 != to) ++field7;

            // leggi settimo campo (c_mktsegment)
            char mktsegment[11] = {0};
            const char* m = field7;
            parseString(m, to, mktsegment, 11, '|');

            // salva il record
            CustomerRecord rec{custkey, {}};
            memcpy(rec.mktsegment, mktsegment, 11);
            localRecords.push_back(rec);

            // avanzare alla fine della riga
            iter = find<'\n'>(iter, to);
            if (iter != to) ++iter;
         }
         
      }));
   }

   // wait for all threads
   for (auto& t : threads) t.join();

   // aggrega i record da tutti i thread
   size_t totalRecords = 0;
   for (const auto& v : recordsPerThread) {
      totalRecords += v.size();
   }
   
   allRecords.reserve(totalRecords);
   for (auto& v : recordsPerThread) {
      allRecords.insert(allRecords.end(), 
                       make_move_iterator(v.begin()), 
                       make_move_iterator(v.end()));
   }

   in.close();
   return totalRecords;
}

// Calcola avg(l_quantity)*100 in parallelo
static size_t computeAvg(const unordered_map<int,int> orderToCustomer, unordered_set<int> setOfCustomerForSegment)
{
   // Definisco le variabili per somma delle quantity...
   double sum_qty = 0.0;
   // ...e per il conteggio dei record che soddisfano le condizioni di join
   size_t match_count = 0;

   // Inizio il calcolo in parallelo
   vector<thread> threads;
   const unsigned numThreads = thread::hardware_concurrency();

   // Calcola la dimensione di ogni blocco
   size_t blockSize = lineitemRecords.size() / numThreads;
   size_t residual = lineitemRecords.size() % numThreads;   

   size_t start=0;
   for (unsigned index=0; index!=numThreads; ++index) {
      // Launch a new thread for each chunk of the file lineitem
      // Each thread computes the avg for its chunk    
      size_t end = start + blockSize + (index < residual ? 1 : 0);

      threads.push_back(thread([index, start, end, orderToCustomer, setOfCustomerForSegment, &sum_qty, &match_count ]() {

            double local_sum_qty = 0.0;
            size_t local_match_count = 0;

            // Itera tutti i record del relativo blocco:
            for (size_t i = start; i < end && i < lineitemRecords.size(); ++i) {

               auto l = lineitemRecords[i];
               // per ogni lineitem cerca l'orderkey del lineitem nella mappa orderToCustomer
               auto orderkey_custkey = orderToCustomer.find(l.orderkey);

               // Se trovato un Customer valido (non ho raggiunto la fine)
               if (orderkey_custkey != orderToCustomer.end()) {
                  
                  // Recupera il custkey associato all'orderkey trovato
                  int custkey = orderkey_custkey->second;

                  // Usa il valore del secondo elemento della coppia chiave-valore della hashmap
                  // (c_custkey dell'iteratore custkey) per cercare nel set il relativo segmento di mercato
                  auto custkeyOfCustomerSegment = setOfCustomerForSegment.find(custkey);

                  // Verifico se nel set segmentCustomer ho un segment (non ho raggiunto la fine)
                  if (custkeyOfCustomerSegment != setOfCustomerForSegment.end()) {
                     // Se arrivati qui, entrambe le condizioni di join sono soddisfatte
                     // Quindi, sommo la quantity presente nel lineitem ed incremento il conteggio
                     local_sum_qty += l.quantity;
                     ++local_match_count;
                  }      
               }
            }

            sum_qty +=local_sum_qty;
            match_count += local_match_count;
         }));

      start = end;
   }

   // wait for all threads
   for (auto& t:threads) 
      t.join();

   size_t result = 0.0;
   if (match_count) 
      result = (sum_qty / static_cast<double>(match_count)) * 100.0;

   return result;
}


//---------------------------------------------------------------------------
JoinQuery::JoinQuery(std::string lineitem, std::string orders, std::string customer)                   
{
   // Verifica che esista il file lineitem
   ifstream ifLineitem(lineitem);
   assert(ifLineitem); 

   // Verifica che esista il file orders
   ifstream ifOrders(orders);
   assert(ifOrders);

   // Verifica che esista il file customer
   ifstream ifCustomer(customer);
   assert(ifCustomer);
   
   // Elabora i tre file in parallelo usando thread separati, uno per ogni file
   thread t1([&]() { processLineItem(lineitem.c_str(), lineitemRecords); });
   thread t2([&]() { processOrders(orders.c_str(), ordersRecords); });
   thread t3([&]() { processCustomer(customer.c_str(), customerRecords); });

   // Sync dei thread
   t1.join();
   t2.join();
   t3.join();

}

//---------------------------------------------------------------------------
size_t JoinQuery::avg(std::string segmentParam)
{
   // STEP 1: Costruisci le strutture dati per la join
   // Costruisce una mappa hash di tipo unordered_map<int,int>
   // che associa ogni orderkey al suo custkey (iterando su ordersRecords). 
   // Vantaggio: lookup O(1) medio ottimo per velocizzare la join.
   unordered_map<int,int> orderToCustomer;
   
   // Riserva spazio per la mappa hash per evitare riallocazioni (e quindi migliorare le prestazioni)
   // durante l'inserimento dei record.
   // Si stima una dimensione iniziale doppia rispetto al numero degli ordini
   orderToCustomer.reserve(ordersRecords.size() ? ordersRecords.size() * 2 : 1);

   // Popola la mappa hash, gestendo ogni coppia (o_orderkey, o_custkey)
   // orderkey è la chiave della mappa, custkey è il valore associato
   for (const auto &o : ordersRecords) 
      orderToCustomer[o.orderkey] = o.custkey;


   // STEP 2: Filtro dei customer in base a segmentParam 
   // Costruisce un set unordered_set<int> contenente tutti i custkey dei customerRecords 
   // il cui campo mktsegment è uguale a segmentParam. 
   // Il confronto è fatto con std::strncmp(c.mktsegment, segmentParam.c_str(), 9) == 0 (confronta fino a 9 caratteri).   
   unordered_set<int> setOfCustomerForSegment;

   // Riserva spazio per il set per evitare riallocazioni durante l'inserimento
   // Si stima una dimensione iniziale doppia rispetto al numero dei customer
   setOfCustomerForSegment.reserve(customerRecords.size() ? customerRecords.size() * 2 : 1);
   
   // Popola il set con i soli custkey dei customer che corrispondono al segmento
   // specificato da segmentParam
   for (const auto &c : customerRecords) {
      if (strncmp(c.mktsegment, segmentParam.c_str(), 10) == 0) 
         // Se il segmento corrisponde, inserisce il custkey nel set
         setOfCustomerForSegment.insert(c.custkey);
   }

   // STEP 3: Calcola avg
   size_t result = computeAvg(orderToCustomer, setOfCustomerForSegment);
   return result;

}
//---------------------------------------------------------------------------
size_t JoinQuery::lineCount(std::string rel)
{
   std::ifstream relation(rel);
   assert(relation);  // make sure the provided string references a file
   size_t n = 0;
   for (std::string line; std::getline(relation, line);) n++;
   return n;
}
//---------------------------------------------------------------------------
