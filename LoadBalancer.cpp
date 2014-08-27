#include "LoadBalancer.hpp"


int LoadBalancer::running_thread_count = 0;
int LoadBalancer::max_threads = 5;
mutex LoadBalancer::global_mutex;


LoadBalancer::LoadBalancer(bool inception):
	inception(inception),
	sleep_time(5) {
}


void LoadBalancer::add_job(function<void(void)> func) {
	todo_list.push(func);
}


void LoadBalancer::start_jobs() {
	if (inception) {
		global_mutex.lock();
		max_threads++;
		global_mutex.unlock();
	}


	DEBUG_DO(cout << "start spinning" << endl);
	while (!(running_threads.empty() && todo_list.empty() && finished_jobs.empty())) {

		// check for finished threads and join them
		while (!finished_jobs.empty()) {
			local_mutex.lock();
			thread::id &finished_id = finished_jobs.front();
			running_threads[finished_id]->join();
			running_threads.erase(finished_id);
			finished_jobs.pop();
			global_mutex.lock();
			running_thread_count--;
			global_mutex.unlock();
			local_mutex.unlock();
			DEBUG_DO(cout << "\tremoved job " << finished_id << endl);
		}

		// start jobs
		while (running_thread_count < max_threads &&  !todo_list.empty()) {
			local_mutex.lock();
			DEBUG_DO(cout << "\tadding job" << endl);
			function<void(void)> &func = todo_list.front();
			thread_ptr tp(new thread(bind(&LoadBalancer::do_job, this, func)));
			running_threads[tp->get_id()] = tp;
			todo_list.pop();
			global_mutex.lock();
			running_thread_count++;
			global_mutex.unlock();
			local_mutex.unlock();
		}

		// sleep
		this_thread::sleep_for(sleep_time);
	}

	if (inception) {
		global_mutex.lock();
		max_threads--;
		global_mutex.unlock();
	}
	DEBUG_DO(cout << "finished" << endl);
}



void LoadBalancer::job_finished(thread::id id) {
	local_mutex.lock();
	finished_jobs.push(id);
	local_mutex.unlock();
}


void LoadBalancer::do_job(function<void(void)> func) {
	DEBUG_DO(cout << "\tstarting job " << endl;);
	func();
	job_finished(this_thread::get_id());
	DEBUG_DO(cout << "\tfinishing job " << this_thread::get_id() << endl);
}
