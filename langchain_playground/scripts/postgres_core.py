import psycopg2
import json

class PostgresCore:
    def __init__(self, reset = True):
        self.init_db(reset)

    def init_db(self, _reset):
        self.db_conn = psycopg2.connect(
            host = "localhost",
            database = "mechllm",
            user = "postgres",
            password = "admin",
            port = 5432
        )

        self.db_cur = self.db_conn.cursor()

        if(_reset):
            self.db_cur.execute(
                """
                    DROP TABLE IF EXISTS objects_map;
                """
            )

            self.db_cur.execute(
                """
                    DROP TABLE IF EXISTS video_summaries;
                """
            )

            self.db_conn.commit()

            self.db_cur.execute(
                """
                    CREATE TABLE IF NOT EXISTS objects_map (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) UNIQUE,
                        features TEXT[],
                        reference_videos INT[][],
                        summary TEXT
                    );
                """
            )

            self.db_cur.execute(
                """
                CREATE TABLE IF NOT EXISTS video_summaries (
                    id SERIAL PRIMARY KEY,
                    file_name VARCHAR(255) UNIQUE,
                    start_time INT NOT NULL,
                    end_time INT NOT NULL,
                    summary TEXT
                );
                """
            )

            self.db_conn.commit()

    def post_objects_map_db(self, _name, _features, _reference_videos, _summary):
        insert_query = """
            INSERT INTO objects_map (name, features, reference_videos, summary) 
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name) 
            DO UPDATE SET 
                features = EXCLUDED.features,
                reference_videos = EXCLUDED.reference_videos
        """

        self.db_cur.execute(insert_query, (_name, _features, _reference_videos, _summary))
        self.db_conn.commit()
    
    def get_objects_map_record_by_name_db(self, name):
        select_query = """
            SELECT * 
            FROM objects_map 
            WHERE name = %s
        """
        self.db_cur.execute(select_query, (name,))
        result = self.db_cur.fetchone()

        if result:
            col_names = [desc[0] for desc in self.db_cur.description]
            return dict(zip(col_names, result))
        else:
            return None
        
    def get_objects_map_name_list_db(self):
        select_query = """
            SELECT name
            FROM objects_map
        """
        self.db_cur.execute(select_query)

        names = [row[0] for row in self.db_cur.fetchall()]

        return names
    
    def get_video_summary_list_db(self):
        select_query = """
            SELECT *
            FROM video_summaries
        """
        self.db_cur.execute(select_query)

        rows = self.db_cur.fetchall()
        # print(rows)
        # data = [dict(row) for row in rows]

        return rows
        
    
    def get_video_summary_record_by_name_db(self, _filename):
        select_query = """
            SELECT * 
            FROM video_summaries 
            WHERE file_name = %s
        """
        self.db_cur.execute(select_query, (_filename,))
        result = self.db_cur.fetchone()

        if result:
            col_names = [desc[0] for desc in self.db_cur.description]
            return dict(zip(col_names, result))
        else:
            return None
        
    def get_video_summary_name_list_db(self):
        select_query = """
            SELECT file_name
            FROM video_summaries
        """
        self.db_cur.execute(select_query)

        names = [row[0] for row in self.db_cur.fetchall()]

        return names

    # def post_video_summaries_db(self, _file_name, _start_time, _end_time, _summary):
    #     insert_query = """
    #         INSERT INTO video_summaries (file_name, start_time, end_time, summary)
    #         VALUES (%s, %s, %s, %s);
    #         """
        
    #     self.db_cur.execute(insert_query, (_file_name, _start_time, _end_time, _summary))

    #     self.db_conn.commit()

    def post_video_record_db(self, _file_name, _start_time, _end_time):
        insert_query = """
            INSERT INTO video_summaries (file_name, start_time, end_time)
            VALUES (%s, %s, %s);
            """
        
        self.db_cur.execute(insert_query, (_file_name, _start_time, _end_time))

        self.db_conn.commit()

    def post_video_summary_db(self, _file_name, _summary):
        insert_query = """
            UPDATE video_summaries
            SET summary = %s
            WHERE file_name = %s
        """
        
        self.db_cur.execute(insert_query, (_summary, _file_name))

        self.db_conn.commit()

if __name__ == "__main__":
    postgres_core = PostgresCore()
    postgres_core.post_objects_map_db("test_key", ["test", "test2"], [["test", "0", "30"], ["tes2t", "10", "40"]])