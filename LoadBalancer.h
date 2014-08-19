/**
 *	TODO: 	* switch from boost to std
**/

#include <boost/thread.hpp>
#include <boost/pending/queue.hpp>
#include <boost/unordered_map.hpp>

using namespace std;

typedef boost::shared_ptr<boost::thread> thread_ptr;



class LoadBalancer {
public:
	LoadBalancer(int max_threads = 4):
		max_threads(max_threads),
		sleep_time(5){};
	~LoadBalancer() {};


	void add_job(boost::function<void(void)> func) {
		todo_list.push(func);
	}

	void start_jobs() {
		cout << "start spinning" << endl;
		while (!(running_threads.empty() && todo_list.empty()  && finished_jobs.empty())) {

			// check for finished threads and join them
			while(!finished_jobs.empty()) {
				mutex.lock();
				boost::thread::id &finished_id = finished_jobs.front();
				running_threads[finished_id]->join();
				running_threads.erase(finished_id);
				finished_jobs.pop();
				mutex.unlock();
				cout << "\tremoved job " << finished_id << endl;
			}

			// start jobs
			while (running_threads.size() < max_threads &&  !todo_list.empty()) {
				mutex.lock();
				cout << "\tadding job" << endl;
				boost::function<void(void)> &func = todo_list.front();
				thread_ptr tp(new boost::thread(boost::bind(&LoadBalancer::do_job, this, func)));
				running_threads[tp->get_id()] = tp;
				todo_list.pop();
				mutex.unlock();
			}

			// sleep
			boost::this_thread::sleep(sleep_time);
		}
		cout << "finished" << endl;
	}


private:
	void job_finished(boost::thread::id id) {
		mutex.lock();
		finished_jobs.push(id);
		mutex.unlock();
	}
	void do_job(boost::function<void(void)> func) {
		cout << "\tstarting job " << boost::this_thread::get_id() << endl;
		func();
		job_finished(boost::this_thread::get_id());
		cout << "\tfinishing job " << boost::this_thread::get_id() << endl;
	}


	int max_threads;
	boost::posix_time::milliseconds sleep_time;
	boost::queue<boost::thread::id> finished_jobs;
	boost::queue<boost::function<void(void)> > todo_list;
	boost::unordered_map<boost::thread::id, thread_ptr> running_threads;
	boost::thread::id default_id;
	boost::mutex mutex;
};